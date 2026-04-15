from __future__ import annotations

import json
import re
from typing import Any

from datasets import Dataset

from dataos.eval.metrics.exact_match import aggregate_exact_match, exact_match
from dataos.eval.types import EvalConfig, EvalRequest, EvalResultBundle, EvalSampleResult, ModelResponse

LABELS = ("a", "b", "c", "d")
OPTION_LABEL_RE = re.compile(r"(?i)(?<![a-z0-9])([a-d])\s*[\)\].:：、]")
LINE_OPTION_LABEL_RE = re.compile(r"(?im)^\s*([a-d])\s*[\)\].:：、]")
DIRECT_LABEL_RE = re.compile(r"(?i)^\s*[\(\[]?\s*([a-d])\s*[\)\].。!！,，;；:]?\s*$")
PREFIXED_LABEL_RE = re.compile(
    r"(?i)(?:answer|ans|choice|option|label|choose|chose|select(?:ed)?|"
    r"答案|正确答案|选项|选择|应选|我选|选)"
    r"\s*(?:is|as|为|是|[:：])?\s*[\(\[]?\s*([a-d])\s*[\)\]]?"
)
STANDALONE_LABEL_RE = re.compile(r"(?i)(?<![a-z])([a-d])(?![a-z])")


class ClimaQATask:
    name = "climaqa"
    version = "v1"
    supported_request_types = {"generate_until"}

    def default_split(self) -> str:
        return "test"

    def build_requests(self, dataset: Dataset, config: EvalConfig) -> list[EvalRequest]:
        del config
        requests: list[EvalRequest] = []
        for index, row in enumerate(dataset):
            question = _extract_question(row, index)
            raw_options = _extract_raw_options(row, index)
            parsed_options = _parse_options(raw_options, index)
            reference = _extract_reference_answer(row, index)
            requests.append(
                EvalRequest(
                    sample_id=str(index),
                    index=index,
                    request_type="generate_until",
                    prompt=_build_prompt(question=question, options=parsed_options),
                    choices=list(LABELS),
                    meta={
                        "complexity": _extract_complexity(row),
                        "raw_options": raw_options,
                        "parsed_options": parsed_options,
                        "reference_answer": reference,
                    },
                )
            )
        return requests

    def score_responses(
        self,
        dataset: Dataset,
        requests: list[EvalRequest],
        responses: list[ModelResponse],
        config: EvalConfig,
    ) -> EvalResultBundle:
        del config
        rows = list(dataset)
        if len(rows) != len(requests):
            raise ValueError("Request count does not match evaluation dataset size")
        if len(rows) != len(responses):
            raise ValueError("Response count does not match evaluation dataset size")

        samples: list[EvalSampleResult] = []
        sample_scores: list[float] = []
        invalid_count = 0
        for index, (row, request, response) in enumerate(zip(rows, requests, responses)):
            raw_options = _extract_raw_options(row, index)
            parsed_options = _parse_options(raw_options, index)
            raw_reference = _extract_raw_reference(row, index)
            normalized_reference = _parse_answer(raw_reference)
            if normalized_reference is None:
                raise ValueError(
                    f"Row {index} requires 'Answer'/'answer' to resolve to one of a/b/c/d"
                )
            normalized_prediction = _parse_prediction(response.text or "")
            score = exact_match(normalized_prediction, normalized_reference)
            sample_scores.append(score)
            invalid_count += int(normalized_prediction is None)
            samples.append(
                EvalSampleResult(
                    sample_id=request.sample_id,
                    index=index,
                    prompt=request.prompt,
                    prediction=response.text,
                    reference=raw_reference,
                    normalized_prediction=normalized_prediction,
                    normalized_reference=normalized_reference,
                    metrics={
                        "accuracy": score,
                        "exact_match": score,
                    },
                    meta={
                        "complexity": _extract_complexity(row),
                        "raw_options": raw_options,
                        "parsed_options": parsed_options,
                        "parsed_prediction": normalized_prediction,
                        "parsed_answer": normalized_reference,
                        "is_invalid": normalized_prediction is None,
                        "is_correct": bool(score),
                    },
                )
            )

        total = len(samples)
        correct = int(sum(sample_scores))
        accuracy = aggregate_exact_match(sample_scores)
        return EvalResultBundle(
            metric_name="accuracy",
            score=accuracy,
            summary={
                "accuracy": accuracy,
                "exact_match": accuracy,
                "correct": correct,
                "num_examples": total,
                "invalid_predictions": invalid_count,
                "invalid_prediction_rate": invalid_count / total if total else 0.0,
            },
            samples=samples,
        )


def _extract_question(row: dict[str, Any], index: int) -> str:
    value = _first_present_field(row, ("Question", "question"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {index} requires a non-empty 'Question'/'question' field")
    return value.strip()


def _extract_raw_options(row: dict[str, Any], index: int) -> Any:
    value = _first_present_field(row, ("Options", "options"))
    if value in (None, ""):
        raise ValueError(f"Row {index} requires a non-empty 'Options'/'options' field")
    return value


def _extract_raw_reference(row: dict[str, Any], index: int) -> Any:
    value = _first_present_field(row, ("Answer", "answer"))
    if value in (None, ""):
        raise ValueError(f"Row {index} requires a non-empty 'Answer'/'answer' field")
    return value


def _extract_reference_answer(row: dict[str, Any], index: int) -> str:
    value = _extract_raw_reference(row, index)
    answer = _parse_answer(value)
    if answer is None:
        raise ValueError(f"Row {index} requires 'Answer'/'answer' to resolve to one of a/b/c/d")
    return answer


def _extract_complexity(row: dict[str, Any]) -> Any:
    return _first_present_field(row, ("Complexity", "complexity"))


def _first_present_field(row: dict[str, Any], field_names: tuple[str, ...]) -> Any:
    for field_name in field_names:
        value = row.get(field_name)
        if value not in (None, ""):
            return value
    return None


def _build_prompt(*, question: str, options: dict[str, str]) -> str:
    option_lines = "\n".join(f"{label}) {options[label]}" for label in LABELS)
    return (
        f"Question:\n{question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Only output one letter: a, b, c, or d."
    )


def _parse_options(raw_options: Any, index: int) -> dict[str, str]:
    if isinstance(raw_options, dict):
        return _validate_options(_parse_mapping_options(raw_options), index)
    if isinstance(raw_options, (list, tuple)):
        return _validate_options(_parse_sequence_options(raw_options), index)
    if isinstance(raw_options, str):
        parsed_json = _parse_json_options(raw_options)
        if parsed_json is not None:
            return _parse_options(parsed_json, index)
        return _validate_options(_parse_string_options(raw_options), index)
    raise ValueError(
        f"Row {index} requires 'Options'/'options' to be a string, list, tuple, or dict"
    )


def _parse_mapping_options(raw_options: dict[Any, Any]) -> dict[str, str]:
    options: dict[str, str] = {}
    for key, value in raw_options.items():
        label = _parse_answer(str(key))
        if label in LABELS:
            options[label] = _coerce_option_text(value)
    return options


def _parse_sequence_options(raw_options: list[Any] | tuple[Any, ...]) -> dict[str, str]:
    options: dict[str, str] = {}
    detected_labels: dict[str, str] = {}
    for value in raw_options:
        text = _coerce_option_text(value)
        match = OPTION_LABEL_RE.match(text)
        if match:
            detected_labels[match.group(1).lower()] = text[match.end() :].strip()
    if set(detected_labels) >= set(LABELS):
        return {label: detected_labels[label] for label in LABELS}

    for label, value in zip(LABELS, raw_options):
        options[label] = _strip_option_prefix(_coerce_option_text(value), label)
    return options


def _parse_string_options(raw_options: str) -> dict[str, str]:
    parts = [part.strip() for part in raw_options.split("--------------------") if part.strip()]
    if len(parts) >= len(LABELS):
        return {
            label: _strip_option_prefix(part, label)
            for label, part in zip(LABELS, parts, strict=False)
        }

    matches = list(LINE_OPTION_LABEL_RE.finditer(raw_options))
    selected: list[re.Match[str]] = []
    expected_index = 0
    for match in matches:
        if expected_index >= len(LABELS):
            break
        if match.group(1).lower() == LABELS[expected_index]:
            selected.append(match)
            expected_index += 1
    if len(selected) == len(LABELS):
        options: dict[str, str] = {}
        for option_index, match in enumerate(selected):
            label = LABELS[option_index]
            next_start = selected[option_index + 1].start() if option_index + 1 < len(selected) else len(raw_options)
            options[label] = raw_options[match.end() : next_start].strip().strip("-|;；。 \n\r\t")
        return options
    return {}


def _parse_json_options(raw_options: str) -> Any | None:
    stripped = raw_options.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _strip_option_prefix(text: str, expected_label: str) -> str:
    match = OPTION_LABEL_RE.match(text)
    if match and match.group(1).lower() == expected_label:
        return text[match.end() :].strip()
    return text.strip()


def _coerce_option_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _validate_options(options: dict[str, str], index: int) -> dict[str, str]:
    missing = [label for label in LABELS if not options.get(label)]
    if missing:
        labels = "/".join(missing)
        raise ValueError(f"Row {index} requires non-empty options for labels: {labels}")
    return {label: options[label].strip() for label in LABELS}


def _parse_prediction(text: str) -> str | None:
    if not text.strip():
        return None
    return _parse_answer(text)


def _parse_answer(value: Any) -> str | None:
    if isinstance(value, int):
        return LABELS[value] if 0 <= value < len(LABELS) else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    direct_match = DIRECT_LABEL_RE.match(text)
    if direct_match:
        return direct_match.group(1).lower()

    prefixed_match = PREFIXED_LABEL_RE.search(text)
    if prefixed_match:
        return prefixed_match.group(1).lower()

    standalone_labels = [match.lower() for match in STANDALONE_LABEL_RE.findall(text)]
    unique_labels = set(standalone_labels)
    if len(unique_labels) == 1:
        return standalone_labels[0]
    return None
