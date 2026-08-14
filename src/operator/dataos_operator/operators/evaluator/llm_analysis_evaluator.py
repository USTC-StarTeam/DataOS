"""Use an LLM to score a sample across five analysis dimensions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from collections.abc import Mapping
from typing import Any

from dataos_operator.core import EvaluatorOperator, Record
from dataos_operator.core.chat_formats import coerce_text, normalize_chat_messages
from dataos_operator.utils.llm_clients import call_llm_json


class LLMAnalysisEvaluator(EvaluatorOperator):
    """Score a sample with an LLM across five dimensions (quality/relevance/informativeness/clarity/safety), writing a normalized score and label.

    Implementation: build the sample text, call the LLM for a five-dimension JSON score, then
    average and normalize to ``[0,1]`` into ``metadata.llm_analysis.*``; retries, falling back to
    a neutral score on failure; API backends default to ``concurrency=8`` parallelism.
    Dependency: online operator; calls an external LLM API by default (OpenAI-compatible, needs
    api_key/base_url), with an optional local ``backend="hf"``.
    """

    DEFAULT_PROMPT_TEMPLATE = """Evaluate the following sample across five dimensions: quality, relevance, informativeness, clarity, and safety. Give each dimension a score from 1 to 5, where 1 is very poor and 5 is excellent. Return JSON only with keys dimension_scores, label, and reason. Use this schema: {{"dimension_scores": {{"quality": 1, "relevance": 1, "informativeness": 1, "clarity": 1, "safety": 1}}, "label": "keep", "reason": "..."}}.

Sample:
{input_text}"""
    DEFAULT_SYSTEM_PROMPT = "You are a strict data quality evaluator. Return JSON only."
    DIMENSION_NAMES = (
        "quality",
        "relevance",
        "informativeness",
        "clarity",
        "safety",
    )

    def __init__(
        self,
        model: str,
        backend: str = "openai_compatible",
        analysis_key: str = "metadata.llm_analysis",
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
            backend=backend,
            analysis_key=analysis_key,
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
        self.backend = backend
        self.analysis_key = analysis_key
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

    def evaluate_dataset(self, data: list[Record]) -> list[Record]:
        if self.backend == "hf" or self.concurrency <= 1 or len(data) <= 1:
            return super().evaluate_dataset(data)
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(data))) as executor:
            return list(executor.map(self.evaluate_record, data))

    def evaluate_record(self, record: Record) -> Record:
        if not self.overwrite and self.get_field(record, f"{self.analysis_key}.score", None) is not None:
            return dict(record)

        input_text = self._build_input_text(record)
        prompt = self.prompt_template.format(input_text=input_text)
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
                dimension_scores = self._normalize_dimension_scores(parsed)
                raw_score = sum(dimension_scores.values()) / len(self.DIMENSION_NAMES)
                score = self._normalize_score(raw_score)
                label = self._label_from_score(score, parsed.get("label"))
                reason = coerce_text(parsed.get("reason", "")).strip()
                break
            except (TypeError, ValueError, KeyError) as exc:
                last_error = exc
        else:
            dimension_scores = {name: 3.0 for name in self.DIMENSION_NAMES}
            raw_score = 3.0
            score = self._normalize_score(raw_score)
            label = "review"
            reason = f"Invalid LLM analysis output after retries: {last_error}" if last_error is not None else "Invalid LLM analysis output after retries"

        updated = self.set_field(record, f"{self.analysis_key}.dimension_scores", dimension_scores)
        updated = self.set_field(updated, f"{self.analysis_key}.score_raw", raw_score)
        updated = self.set_field(updated, f"{self.analysis_key}.score", score)
        updated = self.set_field(updated, f"{self.analysis_key}.label", label)
        updated = self.set_field(updated, f"{self.analysis_key}.reason", reason)
        updated = self.set_field(updated, f"{self.analysis_key}.raw_output", raw_output)
        updated = self.set_field(updated, f"{self.analysis_key}.model", self.model)
        updated = self.set_field(updated, f"{self.analysis_key}.backend", self.backend)
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
            raise ValueError("messages must contain non-empty content for analysis")
        return text

    def _format_template(self, template: str, record: Record) -> str:
        try:
            return template.format_map(_FlatRecordMapping(record))
        except KeyError as exc:
            raise KeyError(f"Template field is missing: {exc.args[0]}") from exc

    def _normalize_dimension_scores(self, parsed: Mapping[str, Any]) -> dict[str, float]:
        raw_dimension_scores = parsed.get("dimension_scores")
        if not isinstance(raw_dimension_scores, Mapping):
            candidate_scores = parsed.get("scores")
            if isinstance(candidate_scores, Mapping):
                raw_dimension_scores = candidate_scores
            else:
                raw_dimension_scores = parsed

        overall_score = self._coerce_numeric_score(parsed.get("score"))
        present_scores = {
            name: numeric
            for name in self.DIMENSION_NAMES
            for numeric in [self._coerce_numeric_score(raw_dimension_scores.get(name))]
            if numeric is not None
        }
        fallback_score = overall_score
        if fallback_score is None and present_scores:
            fallback_score = sum(present_scores.values()) / len(present_scores)
        if fallback_score is None:
            raise ValueError("LLM analysis output must contain dimension scores or an overall score")

        normalized: dict[str, float] = {}
        for name in self.DIMENSION_NAMES:
            score = present_scores.get(name, fallback_score)
            normalized[name] = self._validate_score_range(name, score)
        return normalized

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

    def _validate_score_range(self, name: str, score: float) -> float:
        if score < 1 or score > 5:
            raise ValueError(f"Dimension score {name!r} must be between 1 and 5")
        return score

    def _normalize_score(self, raw_score: float) -> float:
        return max(0.0, min(1.0, (raw_score - 1.0) / 4.0))

    def _label_from_score(self, score: float, raw_label: Any) -> str:
        label = coerce_text(raw_label).strip().lower()
        if label:
            return label
        if score >= 0.75:
            return "keep"
        if score >= 0.5:
            return "review"
        return "drop"


class _FlatRecordMapping(dict[str, Any]):
    def __init__(self, record: Mapping[str, Any]) -> None:
        super().__init__()
        self._record = record

    def __missing__(self, key: str) -> Any:
        if key in self._record:
            return self._record[key]
        raise KeyError(key)
