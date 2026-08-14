"""Use an LLM to score a sample for a target task."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from collections.abc import Mapping
from typing import Any

from dataos_operator.core import EvaluatorOperator, Record
from dataos_operator.core.chat_formats import coerce_text, normalize_chat_messages
from dataos_operator.utils.llm_clients import call_llm_json


class LLMTaskRelevanceEvaluator(EvaluatorOperator):
    """Use an LLM to rate a sample's relevance to a given task_desc (1-5), writing a normalized relevance score and label.

    Implementation: feed task_desc plus the sample text to the LLM for a JSON score, normalize to
    ``[0,1]`` into ``metadata.llm_task_relevance.*``; retries, falling back to a neutral score on
    failure; API backends default to parallelism.
    Dependency: online operator; calls an external LLM API by default (OpenAI-compatible, needs
    api_key/base_url), with an optional local ``backend="hf"``.
    """

    DEFAULT_PROMPT_TEMPLATE = """Target task description:
{task_desc}

Evaluate how relevant the following sample is to the target task. Give a score from 1 to 5, where 1 means irrelevant and 5 means highly relevant. Return JSON only with keys score, label, and reason. Use this schema: {{"score": 1, "label": "relevant", "reason": "..."}}.

Sample:
{input_text}"""
    DEFAULT_SYSTEM_PROMPT = "You are a strict task relevance evaluator. Return JSON only."

    def __init__(
        self,
        model: str,
        task_desc: str,
        backend: str = "openai_compatible",
        relevance_key: str = "metadata.llm_task_relevance",
        messages_key: str = "messages",
        query_key: str = "query",
        answer_key: str = "answer",
        text_key: str = "text",
        input_template: str | None = None,
        prompt_template: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_params: dict[str, Any] | None = None,
        concurrency: int = 8,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            task_desc=task_desc,
            backend=backend,
            relevance_key=relevance_key,
            messages_key=messages_key,
            query_key=query_key,
            answer_key=answer_key,
            text_key=text_key,
            input_template=input_template,
            prompt_template=prompt_template,
            api_key=api_key,
            base_url=base_url,
            model_params=model_params,
            concurrency=concurrency,
            overwrite=overwrite,
            **kwargs,
        )
        self.model = model
        self.task_desc = task_desc.strip()
        self.backend = backend
        self.relevance_key = relevance_key
        self.messages_key = messages_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.text_key = text_key
        self.input_template = input_template
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.api_key = api_key
        self.base_url = base_url
        self.model_params = dict(model_params or {})
        self.concurrency = max(1, int(concurrency))
        self.overwrite = overwrite
        if not self.task_desc:
            raise ValueError("task_desc must not be empty")

    def evaluate_dataset(self, data: list[Record]) -> list[Record]:
        if self.backend == "hf" or self.concurrency <= 1 or len(data) <= 1:
            return super().evaluate_dataset(data)
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(data))) as executor:
            return list(executor.map(self.evaluate_record, data))

    def evaluate_record(self, record: Record) -> Record:
        if not self.overwrite and self.get_field(record, f"{self.relevance_key}.score", None) is not None:
            return dict(record)

        input_text = self._build_input_text(record)
        prompt = self.prompt_template.format(task_desc=self.task_desc, input_text=input_text)
        last_error: Exception | None = None
        for _ in range(3):
            parsed, raw_output = call_llm_json(
                backend=self.backend,
                model=self.model,
                system_prompt=self.DEFAULT_SYSTEM_PROMPT,
                user_prompt=prompt,
                api_key=self.api_key,
                base_url=self.base_url,
                model_params=self.model_params,
            )
            try:
                raw_score = self._parse_raw_score(parsed)
                score = self._normalize_score(raw_score)
                label = self._label_from_score(score, parsed.get("label"))
                reason = coerce_text(parsed.get("reason", "")).strip()
                break
            except (TypeError, ValueError, KeyError) as exc:
                last_error = exc
        else:
            raw_score = 3.0
            score = self._normalize_score(raw_score)
            label = "related"
            reason = f"Invalid LLM task relevance output after retries: {last_error}" if last_error is not None else "Invalid LLM task relevance output after retries"

        updated = self.set_field(record, f"{self.relevance_key}.score_raw", raw_score)
        updated = self.set_field(updated, f"{self.relevance_key}.score", score)
        updated = self.set_field(updated, f"{self.relevance_key}.label", label)
        updated = self.set_field(updated, f"{self.relevance_key}.reason", reason)
        updated = self.set_field(updated, f"{self.relevance_key}.raw_output", raw_output)
        updated = self.set_field(updated, f"{self.relevance_key}.task_desc", self.task_desc)
        updated = self.set_field(updated, f"{self.relevance_key}.model", self.model)
        updated = self.set_field(updated, f"{self.relevance_key}.backend", self.backend)
        return updated

    def _build_input_text(self, record: Record) -> str:
        messages_value = self.get_field(record, self.messages_key, None)
        if messages_value is not None:
            return self._build_input_text_from_messages(messages_value)
        if self.input_template is not None:
            return self._format_template(self.input_template, record).strip()

        query_value = self.get_field(record, self.query_key, None)
        answer_value = self.get_field(record, self.answer_key, None)
        if query_value is not None and answer_value is not None:
            return (
                f"User: {coerce_text(query_value).strip()}\n\n"
                f"Assistant: {coerce_text(answer_value).strip()}"
            )

        text_value = self.get_field(record, self.text_key, None)
        if text_value is not None:
            text = coerce_text(text_value).strip()
            if text:
                return text
        raise KeyError(
            f"Missing input content: expected {self.messages_key!r}, input_template, "
            f"{self.query_key!r}/{self.answer_key!r}, or {self.text_key!r}"
        )

    def _build_input_text_from_messages(self, messages_value: Any) -> str:
        system_message, normalized_messages = normalize_chat_messages(messages_value, strict=True)
        parts: list[str] = []
        if system_message is not None:
            parts.append(f"System: {system_message}")
        for message in normalized_messages:
            role = "User" if message["role"] == "user" else "Assistant"
            parts.append(f"{role}: {message['content']}")
        text = "\n\n".join(parts).strip()
        if not text:
            raise ValueError("messages must contain non-empty content for task relevance")
        return text

    def _format_template(self, template: str, record: Record) -> str:
        try:
            return template.format_map(_FlatRecordMapping(record))
        except KeyError as exc:
            raise KeyError(f"Template field is missing: {exc.args[0]}") from exc

    def _parse_raw_score(self, parsed: Mapping[str, Any]) -> float:
        for key in ("score", "relevance_score", "task_relevance_score"):
            numeric = self._coerce_numeric_score(parsed.get(key))
            if numeric is not None:
                return self._validate_score_range(numeric)

        dimension_scores = parsed.get("dimension_scores") or parsed.get("scores")
        if isinstance(dimension_scores, Mapping):
            values = [
                numeric
                for numeric in (self._coerce_numeric_score(value) for value in dimension_scores.values())
                if numeric is not None
            ]
            if values:
                return self._validate_score_range(sum(values) / len(values))

        raise ValueError("LLM task relevance output must contain a numeric score")

    def _coerce_numeric_score(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

    def _validate_score_range(self, score: float) -> float:
        if score < 1 or score > 5:
            raise ValueError("LLM task relevance score must be between 1 and 5")
        return score

    def _normalize_score(self, raw_score: float) -> float:
        return max(0.0, min(1.0, (raw_score - 1.0) / 4.0))

    def _label_from_score(self, score: float, raw_label: Any) -> str:
        label = coerce_text(raw_label).strip().lower()
        if label:
            return label
        if score >= 0.75:
            return "relevant"
        if score >= 0.5:
            return "related"
        return "irrelevant"


class _FlatRecordMapping(dict[str, Any]):
    def __init__(self, record: Mapping[str, Any]) -> None:
        super().__init__()
        self._record = record

    def __missing__(self, key: str) -> Any:
        if key in self._record:
            return self._record[key]
        raise KeyError(key)
