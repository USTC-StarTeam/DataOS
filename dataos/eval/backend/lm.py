from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from lm_eval.api.model import LM

from dataos.eval.adapters.base import EvalLoglikelihoodAdapter
from dataos.eval.types import (
    AdapterCallContext,
    EvalAdapterConfig,
    EvalRequest,
    EvalRequestConfig,
    ModelLike as ModelArtifact,
)

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


class _GenerateUntilAdapter(Protocol):
    name: str

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: AdapterCallContext,
    ) -> list[Any]: ...


class DataOSLM(LM):
    """Bridge a DataOS adapter (transformers / openai_compatible) into an lm-eval ``LM``.

    Implements only ``generate_until``. Tasks with ``output_type: loglikelihood`` or
    ``multiple_choice`` raise ``NotImplementedError`` until a logprob-capable adapter
    is added.
    """

    def __init__(
        self,
        model_artifact: ModelArtifact,
        adapter: _GenerateUntilAdapter,
        *,
        adapter_config: dict[str, Any] | None = None,
        batch_size: int = 8,
        default_max_new_tokens: int = 256,
        tokenizer_source: str | None = None,
    ) -> None:
        super().__init__()
        self._model = model_artifact
        self._adapter = adapter
        self._adapter_config = dict(adapter_config or {})
        self._batch_size = max(1, int(batch_size))
        self._default_max_new_tokens = max(1, int(default_max_new_tokens))
        self._tokenizer_source = tokenizer_source or model_artifact.meta.get("base_model")
        self._tokenizer_cached: Any | None = None

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_source or self._model.artifact_id

    def _tokenizer(self):
        if self._tokenizer_cached is not None:
            return self._tokenizer_cached
        if not self._tokenizer_source:
            raise ValueError(
                "DataOSLM needs a tokenizer source (model_artifact.meta['base_model'] "
                "or tokenizer_source=) to apply chat templates."
            )
        from transformers import AutoTokenizer

        trust_remote_code = bool(self._adapter_config.get("trust_remote_code", False))
        self._tokenizer_cached = AutoTokenizer.from_pretrained(
            self._tokenizer_source, trust_remote_code=trust_remote_code
        )
        return self._tokenizer_cached

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        tokenizer = self._tokenizer()
        return tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        tokenizer = self._tokenizer()
        return getattr(tokenizer, "chat_template", "") or ""

    def supports_loglikelihood(self) -> bool:
        """True iff the underlying adapter implements ``score_continuations``."""
        return isinstance(self._adapter, EvalLoglikelihoodAdapter)

    def loglikelihood(self, requests):  # noqa: D401 - lm-eval API
        if not isinstance(self._adapter, EvalLoglikelihoodAdapter):
            raise NotImplementedError(
                f"adapter {self._adapter.name!r} does not implement score_continuations; "
                "use a generate-style task variant or pick an adapter with logprob support."
            )
        if not requests:
            return []
        pairs: list[tuple[str, str]] = []
        for instance in requests:
            args = instance.args
            if len(args) < 2:
                raise ValueError(
                    "loglikelihood Instance.args must be (context, continuation); "
                    f"got {len(args)} arg(s)"
                )
            pairs.append((str(args[0]), str(args[1])))
        adapter_config = dict(self._adapter_config)
        config = AdapterCallContext(
            adapter=EvalAdapterConfig(name=self._adapter.name, config=adapter_config),
        )
        return list(self._adapter.score_continuations(self._model, pairs, config))

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "DataOSLM does not support loglikelihood_rolling (perplexity tasks)."
        )

    def generate_until(self, requests: list["Instance"]) -> list[str]:
        if not requests:
            return []
        merged_until, max_new_tokens, do_sample = _merge_gen_kwargs(
            requests, self._default_max_new_tokens
        )
        eval_requests = [
            EvalRequest(
                sample_id=str(instance.idx),
                index=instance.idx,
                request_type="generate_until",
                prompt=instance.args[0],
                stop=list(merged_until[index]) if merged_until[index] else None,
                meta={"task_name": instance.task_name, "doc_id": instance.doc_id},
            )
            for index, instance in enumerate(requests)
        ]
        adapter_config = dict(self._adapter_config)
        adapter_config.setdefault("chat_template_already_applied", True)
        config = AdapterCallContext(
            adapter=EvalAdapterConfig(name=self._adapter.name, config=adapter_config),
            request=EvalRequestConfig(
                type="generate_until",
                batch_size=self._batch_size,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            ),
        )
        responses = self._adapter.generate_until(self._model, eval_requests, config)
        return [_truncate_at_stop(resp.text or "", merged_until[i]) for i, resp in enumerate(responses)]


def _merge_gen_kwargs(
    requests: list["Instance"], default_max_new_tokens: int
) -> tuple[list[list[str]], int, bool]:
    """lm-eval passes ``(context, gen_kwargs)`` per Instance. Collect per-request
    ``until`` lists and a single batch-level ``max_new_tokens``/``do_sample``.
    Adapters today only support uniform generation params, so we take the max of
    per-request ``max_gen_toks`` and OR of ``do_sample``.
    """
    until_per_request: list[list[str]] = []
    max_new_tokens = default_max_new_tokens
    do_sample = False
    for instance in requests:
        kwargs: dict[str, Any] = {}
        if len(instance.args) >= 2 and isinstance(instance.args[1], dict):
            kwargs = instance.args[1]
        until_value = kwargs.get("until") or []
        if isinstance(until_value, str):
            until_value = [until_value]
        until_per_request.append([str(item) for item in until_value])
        if "max_gen_toks" in kwargs and kwargs["max_gen_toks"]:
            max_new_tokens = max(max_new_tokens, int(kwargs["max_gen_toks"]))
        if kwargs.get("do_sample"):
            do_sample = True
    return until_per_request, max_new_tokens, do_sample


def _truncate_at_stop(text: str, stops: list[str]) -> str:
    cut = len(text)
    for stop in stops:
        if not stop:
            continue
        position = text.find(stop)
        if 0 <= position < cut:
            cut = position
    return text[:cut]
