from __future__ import annotations

import asyncio
import copy
import json
import os
from typing import Any

from aiohttp import ClientSession, ClientTimeout, TCPConnector
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import eval_logger
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.models.utils import chunks
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio


@register_model("dataos-local-tool-chat-completions")
class DataOSLocalToolChatCompletion(LocalChatCompletion):
    """OpenAI-compatible chat backend that preserves tool calls for lm-eval tasks.

    lm-eval's built-in ``local-chat-completions`` backend returns only
    ``message.content``. BFCL-style evaluations need the structured
    ``message.tool_calls`` payload produced by vLLM's OpenAI-compatible server.
    This backend keeps normal text generations unchanged and serializes tool
    calls as JSON when present.
    """

    @property
    def api_key(self) -> str:
        return os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

    async def amodel_call(
        self,
        session: ClientSession,
        sem: asyncio.Semaphore,
        messages: Any,
        *,
        generate: bool = True,
        cache_keys: list | None = None,
        ctxlens: list[int] | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[str] | list[tuple[float, bool]] | None:
        """Call a local chat-completions API with safer error logging.

        The lm-eval version used by our server logs ``outputs`` before it is
        assigned when a request fails before a JSON response is produced. That
        masks the real error as ``UnboundLocalError`` and can close the shared
        aiohttp session early during long AIME runs.
        """

        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        outputs: Any = None
        acquired = await sem.acquire()
        try:
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    eval_logger.warning(
                        "API request failed! Status code: %s, Response text: %s. Retrying...",
                        response.status,
                        error_text,
                    )
                response.raise_for_status()
                outputs = await response.json()
            answers = (
                self.parse_generations(outputs=outputs)
                if generate
                else self.parse_logprobs(outputs=outputs, tokens=messages, ctxlens=ctxlens)
            )
            if cache_keys:
                for result, cache in zip(answers, cache_keys, strict=False):
                    self.cache_hook.add_partial(cache_method, cache, result)
            return answers
        except BaseException as exc:
            eval_logger.error("Exception:%r, outputs=%r, retrying.", exc, outputs)
            raise
        finally:
            if acquired:
                sem.release()

    def _create_payload(
        self,
        messages: list[dict[str, Any]],
        generate: bool = False,
        gen_kwargs: dict[str, Any] | None = None,
        seed: int = 1234,
        eos: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        gen_kwargs = gen_kwargs or {}
        official_chat_payload = bool(gen_kwargs.pop("dataos_official_chat_payload", False))
        omit_seed = bool(gen_kwargs.pop("dataos_omit_seed", False))
        if not official_chat_payload:
            payload = super()._create_payload(
                messages,
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=seed,
                eos=eos,
                **kwargs,
            )
            if omit_seed:
                payload.pop("seed", None)
            return payload

        if not generate:
            payload = super()._create_payload(
                messages,
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=seed,
                eos=eos,
                **kwargs,
            )
            if omit_seed:
                payload.pop("seed", None)
            return payload
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise TypeError("official chat payload expects messages as list[dict]")

        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)

        # PostTrainBench official chat evaluators such as HealthBench and
        # ArenaHard-Writing send only model/messages/max_tokens to the
        # OpenAI-compatible endpoint. Do not add lm-eval's default temperature,
        # seed, or stop fields when this compatibility flag is set.
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            **gen_kwargs,
        }

    async def get_batched_requests(
        self,
        requests: list,
        cache_keys: list,
        *,
        generate: bool = True,
        ctxlens: list[int] | None = None,
        **kwargs: Any,
    ) -> list[list[str]] | list[list[tuple[float, bool]]]:
        """Process API requests in bounded windows for long PostTrainBench tasks."""

        ctxlens = ctxlens if ctxlens else [None] * len(requests)
        request_batches = list(
            zip(
                chunks(requests, n=self._batch_size),
                chunks(cache_keys, n=self._batch_size),
                chunks(ctxlens, n=self._batch_size),
                strict=True,
            )
        )
        window_size = max(1, int(os.environ.get("DATAOS_LMEVAL_API_WINDOW", "64")))
        connector = TCPConnector(limit=self._concurrent, ssl=self.verify_certificate)
        sem = asyncio.Semaphore(self._concurrent)
        outputs: list[Any] = []
        async with ClientSession(
            connector=connector,
            timeout=ClientTimeout(total=self.timeout),
        ) as session:
            retry_call = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                reraise=True,
                before_sleep=lambda retry_state: eval_logger.info(
                    "Retry attempt %s", retry_state.attempt_number
                ),
            )(self.amodel_call)
            for start in range(0, len(request_batches), window_size):
                window = request_batches[start : start + window_size]
                tasks = [
                    asyncio.create_task(
                        retry_call(
                            session=session,
                            sem=sem,
                            messages=message,
                            cache_keys=cache_key,
                            generate=generate,
                            ctxlens=ctxlen,
                            **kwargs,
                        )
                    )
                    for message, cache_key, ctxlen in window
                ]
                outputs.extend(await tqdm_asyncio.gather(*tasks, desc="Requesting API"))
        return outputs

    @staticmethod
    def parse_generations(outputs: Any, **_: Any) -> list[str]:
        if not isinstance(outputs, list):
            outputs = [outputs]

        generations: list[str] = []
        for output in outputs:
            choices = output.get("choices", []) if isinstance(output, dict) else []
            if not choices:
                generations.append("")
                continue

            by_index: list[str | None] = [None] * len(choices)
            for choice in choices:
                index = int(choice.get("index", 0))
                message = choice.get("message") or {}
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    by_index[index] = json.dumps(
                        {"tool_calls": tool_calls},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                else:
                    by_index[index] = message.get("content") or ""
            generations.extend(item or "" for item in by_index)
        return generations
