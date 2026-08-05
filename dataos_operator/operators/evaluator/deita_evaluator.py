"""使用 DEITA 风格提示词为样本计算 complexity、quality 与 score 的 evaluator。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dataos_operator.core import EvaluatorOperator, Record
from dataos_operator.core.chat_formats import (
    coerce_text,
    format_llamafactory_alpaca,
    normalize_chat_messages,
)


class DeitaEvaluator(EvaluatorOperator):
    """Compute DEITA-style Evol Complexity, Evol Quality, and their product score.

    Implementation: with the complexity / quality scorer models, take a softmax-weighted
    expected score over the digit tokens 1-6 at the next-token position of the scoring template;
    complexity * quality is the total, written to ``metadata.deita.{complexity,quality,score}``;
    reused when ``overwrite=False``.
    Dependency: local-model operator; loads two HF scorer models on demand (transformers/torch).
    """

    COMPLEXITY_TEMPLATE = (
        "You are a helpful assistant. Please identify the complexity score of the "
        "following user query.\n##Query: {instruction}\n##Complexity: "
    )
    QUALITY_TEMPLATE = (
        "You are a helpful assistant. Please identify the quality score of the "
        "Response corresponding to the Question.\n#Question#:\n{instruction}\n"
        "#Response#:\n{response}\n##Quality: "
    )
    SCORE_CANDIDATES = (1, 2, 3, 4, 5, 6)

    def __init__(
        self,
        complexity_model: str = "hkust-nlp/deita-complexity-scorer",
        quality_model: str = "hkust-nlp/deita-quality-scorer",
        model_params: dict[str, Any] | None = None,
        deita_key: str = "metadata.deita",
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
            complexity_model=complexity_model,
            quality_model=quality_model,
            model_params=model_params,
            deita_key=deita_key,
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
        self.complexity_model = complexity_model
        self.quality_model = quality_model
        self.model_params = dict(model_params or {})
        self.deita_key = deita_key
        self.query_template = query_template
        self.response_template = response_template
        self.messages_key = messages_key
        self.system_key = system_key
        self.text_key = text_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.overwrite = overwrite
        self._torch: Any | None = None
        self._complexity_tokenizer: Any | None = None
        self._complexity_model_instance: Any | None = None
        self._quality_tokenizer: Any | None = None
        self._quality_model_instance: Any | None = None

    def evaluate_record(self, record: Record) -> Record:
        score_path = f"{self.deita_key}.score"
        if not self.overwrite and self.get_field(record, score_path, None) is not None:
            return dict(record)

        instruction_text, response_text = self._instruction_and_response(record)
        complexity = self._compute_complexity(instruction_text)
        quality = self._compute_quality(instruction_text, response_text)
        score = complexity * quality

        updated = self.set_field(record, f"{self.deita_key}.complexity", complexity)
        updated = self.set_field(updated, f"{self.deita_key}.quality", quality)
        updated = self.set_field(updated, f"{self.deita_key}.score", score)
        return updated

    def _instruction_and_response(self, record: Record) -> tuple[str, str]:
        messages_value = self.get_field(record, self.messages_key, None)
        if messages_value is not None:
            return self._instruction_and_response_from_messages(record, messages_value)
        return self._instruction_and_response_from_fields(record)

    def _instruction_and_response_from_messages(
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
        instruction_parts: list[str] = []
        if resolved_system is not None:
            instruction_parts.append(f"System: {coerce_text(resolved_system)}")

        for message in normalized_messages[:-1]:
            role = "User" if message["role"] == "user" else "Assistant"
            instruction_parts.append(f"{role}: {message['content']}")

        instruction_text = "\n\n".join(instruction_parts)
        return instruction_text, last_message["content"]

    def _instruction_and_response_from_fields(self, record: Record) -> tuple[str, str]:
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

        instruction_parts: list[str] = []
        if system_value is not None:
            instruction_parts.append(f"System: {coerce_text(system_value)}")
        instruction_parts.append(format_llamafactory_alpaca(query_text))
        return "\n\n".join(instruction_parts), response_text

    def _format_template(self, template: str, record: Record) -> str:
        try:
            return template.format_map(_FlatRecordMapping(record))
        except KeyError as exc:
            raise KeyError(f"Template field is missing: {exc.args[0]}") from exc

    def _compute_complexity(self, instruction_text: str) -> float:
        if not instruction_text:
            raise ValueError("instruction text must not be empty when computing DEITA complexity")
        prompt = self.COMPLEXITY_TEMPLATE.format(instruction=instruction_text)
        return self._expected_score(prompt, scorer_kind="complexity")

    def _compute_quality(self, instruction_text: str, response_text: str) -> float:
        if not instruction_text:
            raise ValueError("instruction text must not be empty when computing DEITA quality")
        if not response_text:
            raise ValueError("response text must not be empty when computing DEITA quality")
        prompt = self.QUALITY_TEMPLATE.format(
            instruction=instruction_text,
            response=response_text,
        )
        return self._expected_score(prompt, scorer_kind="quality")

    def _expected_score(self, prompt_text: str, *, scorer_kind: str) -> float:
        tokenizer, model = self._get_scorer_components(scorer_kind)
        assert self._torch is not None

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = next(model.parameters()).device
        with self._torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
        next_token_logits = outputs.logits[0, -1]
        score_token_ids = self._score_token_ids(tokenizer)
        score_logits = self._torch.stack([next_token_logits[token_id] for token_id in score_token_ids])
        score_probs = self._torch.softmax(score_logits, dim=0)
        score_values = self._torch.tensor(
            self.SCORE_CANDIDATES,
            dtype=score_probs.dtype,
            device=score_probs.device,
        )
        return float((score_probs * score_values).sum().detach().cpu().item())

    def _get_scorer_components(self, scorer_kind: str) -> tuple[Any, Any]:
        self._ensure_models_loaded()
        if scorer_kind == "complexity":
            assert self._complexity_tokenizer is not None
            assert self._complexity_model_instance is not None
            return self._complexity_tokenizer, self._complexity_model_instance
        if scorer_kind == "quality":
            assert self._quality_tokenizer is not None
            assert self._quality_model_instance is not None
            return self._quality_tokenizer, self._quality_model_instance
        raise ValueError(f"Unsupported scorer kind: {scorer_kind!r}")

    def _ensure_models_loaded(self) -> None:
        if (
            self._complexity_model_instance is not None
            and self._complexity_tokenizer is not None
            and self._quality_model_instance is not None
            and self._quality_tokenizer is not None
            and self._torch is not None
        ):
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        tokenizer_kwargs = {
            key: value
            for key, value in self.model_params.items()
            if key in {"trust_remote_code", "revision", "use_fast", "token", "local_files_only"}
        }
        model_kwargs = dict(self.model_params)

        self._complexity_tokenizer = AutoTokenizer.from_pretrained(
            self.complexity_model,
            **tokenizer_kwargs,
        )
        self._quality_tokenizer = AutoTokenizer.from_pretrained(
            self.quality_model,
            **tokenizer_kwargs,
        )
        for tokenizer in (self._complexity_tokenizer, self._quality_tokenizer):
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

        self._complexity_model_instance = AutoModelForCausalLM.from_pretrained(
            self.complexity_model,
            **model_kwargs,
        )
        self._quality_model_instance = AutoModelForCausalLM.from_pretrained(
            self.quality_model,
            **model_kwargs,
        )
        if "device_map" not in model_kwargs:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._complexity_model_instance.to(device)
            self._quality_model_instance.to(device)
        self._complexity_model_instance.eval()
        self._quality_model_instance.eval()

    def _score_token_ids(self, tokenizer: Any) -> list[int]:
        token_ids: list[int] = []
        for score in self.SCORE_CANDIDATES:
            token_ids.append(self._single_token_id(tokenizer, str(score)))
        return token_ids

    def _single_token_id(self, tokenizer: Any, text: str) -> int:
        direct_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(direct_ids) == 1:
            return int(direct_ids[0])

        spaced_ids = tokenizer.encode(f" {text}", add_special_tokens=False)
        if len(spaced_ids) == 1:
            return int(spaced_ids[0])

        raise ValueError(f"Unable to resolve a single score token for {text!r}")


class _FlatRecordMapping(dict[str, Any]):
    def __init__(self, record: Mapping[str, Any]) -> None:
        super().__init__()
        self._record = record

    def __missing__(self, key: str) -> Any:
        if key in self._record:
            return self._record[key]
        raise KeyError(key)
