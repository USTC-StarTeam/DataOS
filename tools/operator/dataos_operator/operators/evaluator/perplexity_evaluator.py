"""使用语言模型计算样本 response perplexity 的 evaluator。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dataos_operator.core import EvaluatorOperator, Record
from dataos_operator.core.chat_formats import (
    coerce_text,
    format_llamafactory_alpaca,
    normalize_chat_messages,
)


class PerplexityEvaluator(EvaluatorOperator):
    """Compute a sample response's conditional perplexity with a causal LM, written to a score field.

    Implementation: prefer ``messages`` to build the context, otherwise assemble query+response
    from fields/templates; compute the conditional cross-entropy over only the response part and
    take ``exp`` to get perplexity, written to ``score_key`` (default ``metadata.perplexity``);
    reuse an existing score when ``overwrite=False``.
    Dependency: local-model operator; loads a HF causal LM on demand (transformers/torch).
    """

    def __init__(
        self,
        hf_model: str,
        model_params: dict[str, Any] | None = None,
        score_key: str = "metadata.perplexity",
        query_template: str | None = None,
        response_template: str | None = None,
        messages_key: str = "messages",
        system_key: str = "system",
        text_key: str = "text",
        query_key: str = "query",
        answer_key: str = "answer",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            hf_model=hf_model,
            model_params=model_params,
            score_key=score_key,
            query_template=query_template,
            response_template=response_template,
            messages_key=messages_key,
            system_key=system_key,
            text_key=text_key,
            query_key=query_key,
            answer_key=answer_key,
            overwrite=overwrite,
            **kwargs,
        )
        self.hf_model = hf_model
        self.model_params = dict(model_params or {})
        self.score_key = score_key
        self.query_template = query_template
        self.response_template = response_template
        self.messages_key = messages_key
        self.system_key = system_key
        self.text_key = text_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.overwrite = overwrite
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def evaluate_record(self, record: Record) -> Record:
        if not self.overwrite and self.get_field(record, self.score_key, None) is not None:
            return dict(record)

        prompt_text, response_text = self._prompt_and_response(record)
        score = self._compute_perplexity(prompt_text, response_text)
        return self.set_field(record, self.score_key, score)

    def _prompt_and_response(self, record: Record) -> tuple[str, str]:
        messages_value = self.get_field(record, self.messages_key, None)
        if messages_value is not None:
            return self._prompt_and_response_from_messages(record, messages_value)
        return self._prompt_and_response_from_fields(record)

    def _prompt_and_response_from_messages(
        self,
        record: Record,
        messages_value: Any,
    ) -> tuple[str, str]:
        system_from_messages, normalized_messages = normalize_chat_messages(
            messages_value,
            strict=True,
        )
        if not normalized_messages:
            raise ValueError("messages must contain at least one non-system message")

        last_message = normalized_messages[-1]
        if last_message["role"] != "assistant":
            raise ValueError("messages must end with an assistant response")

        explicit_system = self.get_field(record, self.system_key, None)
        resolved_system = explicit_system if explicit_system is not None else system_from_messages
        prompt_parts: list[str] = []
        if resolved_system is not None:
            prompt_parts.append(f"System: {coerce_text(resolved_system)}")

        for message in normalized_messages[:-1]:
            role = "User" if message["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {message['content']}")

        prompt_text = "\n\n".join(prompt_parts)
        if prompt_text:
            prompt_text = f"{prompt_text}\n\nAssistant: "
        else:
            prompt_text = "Assistant: "
        return prompt_text, last_message["content"]

    def _prompt_and_response_from_fields(self, record: Record) -> tuple[str, str]:
        system_value = self.get_field(record, self.system_key, None)
        if self.query_template is not None:
            query_text = self._format_template(self.query_template, record)
        else:
            query_value = self.get_field(record, self.query_key, None)
            if query_value is None:
                query_value = self.get_field(record, self.text_key, None)
            if query_value is None:
                raise KeyError(
                    f"Missing query content: expected {self.query_key!r} or {self.text_key!r}"
                )
            query_text = coerce_text(query_value)

        if self.response_template is not None:
            response_text = self._format_template(self.response_template, record)
        else:
            answer_value = self.get_field(record, self.answer_key, None)
            if answer_value is None:
                raise KeyError(f"Missing response content: expected {self.answer_key!r}")
            response_text = coerce_text(answer_value)

        prompt_parts: list[str] = []
        if system_value is not None:
            prompt_parts.append(f"System: {coerce_text(system_value)}")
        prompt_parts.append(format_llamafactory_alpaca(query_text))
        prompt_text = "\n\n".join(prompt_parts) + "\n\nAssistant: "
        return prompt_text, response_text

    def _format_template(self, template: str, record: Record) -> str:
        try:
            return template.format_map(_FlatRecordMapping(record))
        except KeyError as exc:
            raise KeyError(f"Template field is missing: {exc.args[0]}") from exc

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        tokenizer_kwargs = {
            key: value
            for key, value in self.model_params.items()
            if key in {"trust_remote_code", "revision", "use_fast", "token", "local_files_only"}
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model, **tokenizer_kwargs)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = dict(self.model_params)
        self._model = AutoModelForCausalLM.from_pretrained(self.hf_model, **model_kwargs)
        if "device_map" not in model_kwargs:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
        self._model.eval()

    def _encode_prompt_and_response(self, prompt_text: str, response_text: str) -> tuple[Any, Any, int]:
        self._ensure_model_loaded()
        assert self._tokenizer is not None
        assert self._torch is not None

        prompt_tokens = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        response_tokens = self._tokenizer(
            response_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        prompt_ids = prompt_tokens["input_ids"]
        response_ids = response_tokens["input_ids"]
        if response_ids.shape[1] == 0:
            raise ValueError("response text produced no additional tokens")

        prompt_attention = prompt_tokens.get("attention_mask")
        response_attention = response_tokens.get("attention_mask")
        if prompt_attention is None:
            prompt_attention = self._torch.ones_like(prompt_ids)
        if response_attention is None:
            response_attention = self._torch.ones_like(response_ids)

        input_ids = self._torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = self._torch.cat([prompt_attention, response_attention], dim=1)
        return input_ids, attention_mask, int(prompt_ids.shape[1])

    def _compute_perplexity(self, prompt_text: str, response_text: str) -> float:
        if not response_text:
            raise ValueError("response text must not be empty when computing perplexity")

        self._ensure_model_loaded()
        assert self._model is not None
        assert self._torch is not None

        input_ids, attention_mask, prompt_length = self._encode_prompt_and_response(
            prompt_text,
            response_text,
        )

        labels = input_ids.clone()
        labels[:, :prompt_length] = -100

        device = next(self._model.parameters()).device
        with self._torch.no_grad():
            outputs = self._model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
        return float(self._torch.exp(outputs.loss).detach().cpu().item())


class _FlatRecordMapping(dict[str, Any]):
    def __init__(self, record: Mapping[str, Any]) -> None:
        super().__init__()
        self._record = record

    def __missing__(self, key: str) -> Any:
        if key in self._record:
            return self._record[key]
        raise KeyError(key)
