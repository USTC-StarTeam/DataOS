from __future__ import annotations

import os.path as osp
import re
import string
import unicodedata
from typing import Any

from datasets import Dataset

from dataos.eval.metrics.exact_match import aggregate_exact_match
from dataos.eval.types import EvalConfig, EvalRequest, EvalResultBundle, EvalSampleResult, ModelResponse

DEFAULT_QUESTION_FIELD = "question"
DEFAULT_ANSWER_FIELD = "answer"
DEFAULT_TASK_TYPE_FIELD = "task_type"
OPTION_LABELS = ("A", "B", "C", "D", "E")

MISSING_REFERENCE_VALUES = {"", "nan", "none", "null", "n/a", "na"}
OPENCOMPASS_CHOICE_TASKS = {
    "data_inspection",
    "emotion_identification",
    "entity_disambiguation",
    "financial_facts",
    "financial_terminology",
    "intent_understanding",
    "metric_calculation",
    "value_extraction",
}
OPENCOMPASS_EXTRACTION_TASKS = {"entity_recognition"}
OPENFINDATA_QA_TASKS = {"business_compliance", "security_compliance"}
SOURCE_NAME_FIELDS = (
    "subset",
    "subset_name",
    "source_task",
    "source_name",
    "name",
    "source_file",
)

CHOICE_MARKER_RE = re.compile(
    r"(?:^|\s)(?P<label>[A-Z])\s*[\.\uff0e\u3001:：\)]\s*(?P<text>.*?)(?="
    r"(?:\s+[A-Z]\s*[\.\uff0e\u3001:：\)]\s*)|$)",
    re.DOTALL | re.IGNORECASE,
)
PAREN_CHOICE_MARKER_RE = re.compile(
    r"(?:^|\s)[\(（](?P<label>[A-Z])[\)）]\s*(?P<text>.*?)(?="
    r"(?:\s*[\(（][A-Z][\)）]\s*)|$)",
    re.DOTALL | re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")
ENTITY_SEPARATORS_RE = re.compile(r"[\n\r\t,，、;；|/]+")
BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[\.\)]|[A-Z][\.\)])\s*")
MARKDOWN_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
FINAL_ANSWER_LETTER_RE = re.compile(
    r"(?i)(?:最终答案|最终选项|正确答案|正确选项|答案|选项|故选|应选|选择|answer|final answer)"
    r"\s*(?:is|为|是|:|：)?\s*[\(\[（]?\s*([A-Z])\s*[\)\]）]?"
)
STANDALONE_CHOICE_RE = re.compile(r"^[\(\[（]?\s*([A-Z])\s*[\)\]）]?[\.\uff0e\u3001:：]?$", re.IGNORECASE)


class OpenFinDataTask:
    name = "openfindata"
    version = "v3"
    supported_request_types = {"generate_until"}

    def default_split(self) -> str:
        return "test"

    def build_requests(self, dataset: Dataset, config: EvalConfig) -> list[EvalRequest]:
        task_config = _task_config(config)
        question_field = _field_name(task_config, "question_field", DEFAULT_QUESTION_FIELD)
        answer_field = _field_name(task_config, "answer_field", DEFAULT_ANSWER_FIELD)
        task_type_field = _field_name(task_config, "task_type_field", DEFAULT_TASK_TYPE_FIELD)

        requests: list[EvalRequest] = []
        for index, row in enumerate(dataset):
            question = _require_string_field(row, question_field, index)
            answer = _coerce_answer(row, answer_field, index)
            source_name = _extract_source_name(row)
            output_type = _infer_output_type(
                row.get(task_type_field),
                question,
                answer,
                source_name=source_name,
            )
            choice_options = _extract_row_options(row) if output_type == "choice" else {}
            requests.append(
                EvalRequest(
                    sample_id=str(index),
                    index=index,
                    request_type="generate_until",
                    prompt=_build_prompt(
                        question=question,
                        output_type=output_type,
                        choice_options=choice_options,
                    ),
                    meta={
                        "question_field": question_field,
                        "answer_field": answer_field,
                        "task_type_field": task_type_field,
                        "output_type": output_type,
                        "source_name": source_name,
                        "choice_options": choice_options,
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
        rows = list(dataset)
        if len(rows) != len(requests):
            raise ValueError("Request count does not match evaluation dataset size")
        if len(rows) != len(responses):
            raise ValueError("Response count does not match evaluation dataset size")

        task_config = _task_config(config)
        answer_field = _field_name(task_config, "answer_field", DEFAULT_ANSWER_FIELD)
        task_type_field = _field_name(task_config, "task_type_field", DEFAULT_TASK_TYPE_FIELD)

        samples: list[EvalSampleResult] = []
        sample_scores: list[float] = []
        type_counts: dict[str, int] = {}
        type_correct: dict[str, int] = {}
        type_scored: dict[str, int] = {}
        type_skipped: dict[str, int] = {}
        skipped_count = 0

        for index, (row, request, response) in enumerate(zip(rows, requests, responses)):
            question = _require_string_field(row, _field_name(task_config, "question_field", DEFAULT_QUESTION_FIELD), index)
            reference = _coerce_answer(row, answer_field, index)
            prediction = response.text or ""
            source_name = _extract_source_name(row)
            output_type = _infer_output_type(
                row.get(task_type_field),
                question,
                reference,
                source_name=source_name,
            )
            choice_options = _coerce_choice_options(request.meta.get("choice_options"))
            if not choice_options:
                choice_options = _extract_row_options(row)
            type_counts[output_type] = type_counts.get(output_type, 0) + 1

            if _is_missing_reference(reference):
                skipped_count += 1
                type_skipped[output_type] = type_skipped.get(output_type, 0) + 1
                samples.append(
                    EvalSampleResult(
                        sample_id=request.sample_id,
                        index=index,
                        prompt=request.prompt,
                        prediction=response.text,
                        reference=reference,
                        normalized_prediction=None,
                        normalized_reference=None,
                        metrics={},
                        meta={
                            "output_type": output_type,
                            "is_correct": None,
                            "is_skipped": True,
                            "skip_reason": "missing_reference_answer",
                        },
                    )
                )
                continue

            score, normalized_prediction, normalized_reference = _score_prediction(
                prediction=prediction,
                reference=reference,
                question=question,
                output_type=output_type,
                choice_options=choice_options,
            )
            sample_scores.append(score)
            type_scored[output_type] = type_scored.get(output_type, 0) + 1
            type_correct[output_type] = type_correct.get(output_type, 0) + int(score)
            samples.append(
                EvalSampleResult(
                    sample_id=request.sample_id,
                    index=index,
                    prompt=request.prompt,
                    prediction=response.text,
                    reference=reference,
                    normalized_prediction=normalized_prediction,
                    normalized_reference=normalized_reference,
                    metrics={"exact_match": score},
                    meta={
                        "output_type": output_type,
                        "is_correct": bool(score),
                        "is_skipped": False,
                    },
                )
            )

        scored_count = len(sample_scores)
        correct = int(sum(sample_scores))
        summary: dict[str, Any] = {
            "correct": correct,
            "num_examples": len(samples),
            "num_scored": scored_count,
            "num_skipped": skipped_count,
            "type_counts": type_counts,
            "type_scored": type_scored,
            "type_skipped": type_skipped,
            "type_correct": type_correct,
        }
        for output_type, count in type_scored.items():
            summary[f"{output_type}_accuracy"] = type_correct.get(output_type, 0) / count if count else 0.0

        return EvalResultBundle(
            metric_name="exact_match",
            score=aggregate_exact_match(sample_scores),
            summary=summary,
            samples=samples,
        )


def _task_config(config: EvalConfig) -> dict[str, Any]:
    if config.task is None:
        return {}
    return dict(config.task.config)


def _field_name(task_config: dict[str, Any], key: str, default: str) -> str:
    value = task_config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"eval_config.task.config.{key} must be a non-empty string")
    return value.strip()


def _require_string_field(row: dict[str, Any], field_name: str, index: int) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {index} requires a non-empty '{field_name}' field for openfindata eval")
    return value.strip()


def _coerce_answer(row: dict[str, Any], field_name: str, index: int) -> str:
    if field_name not in row:
        raise ValueError(f"Row {index} requires '{field_name}' field for openfindata eval")
    value = row.get(field_name)
    if isinstance(value, (list, tuple, set)):
        answer = "，".join(str(item) for item in value)
    else:
        answer = "" if value is None else str(value)
    return answer.strip()


def _is_missing_reference(reference: str) -> bool:
    return _normalize_text(reference) in MISSING_REFERENCE_VALUES


def _build_prompt(*, question: str, output_type: str, choice_options: dict[str, str]) -> str:
    prompt_lines = [question.rstrip()]
    if output_type == "choice":
        for label in OPTION_LABELS:
            option_text = choice_options.get(label)
            if option_text:
                prompt_lines.append(f"{label}. {option_text}")
    return "\n".join(prompt_lines).rstrip() + "\n答案: "

def _infer_output_type(task_type: Any, question: str, answer: Any, *, source_name: str) -> str:
    task_text = _normalize_task_type(task_type)
    if source_name in OPENCOMPASS_CHOICE_TASKS:
        return "choice"
    if source_name in OPENCOMPASS_EXTRACTION_TASKS:
        return "extraction"
    if source_name in OPENFINDATA_QA_TASKS:
        return "qa"
    if _looks_like_choice_type(task_text) or (_extract_options(question) and _looks_like_choice_answer(answer)):
        return "choice"
    if _looks_like_extraction_type(task_text, question):
        return "extraction"
    return "qa"


def _normalize_task_type(task_type: Any) -> str:
    if task_type is None:
        return ""
    if isinstance(task_type, (list, tuple, set)):
        text = " ".join(str(item) for item in task_type)
    else:
        text = str(task_type)
    return _normalize_text(text)


def _looks_like_choice_type(task_text: str) -> bool:
    choice_markers = (
        "choice",
        "multiple choice",
        "single choice",
        "select",
        "selection",
        "classification",
        "label",
        "判断",
        "判别",
        "选择",
        "单选",
        "多选",
    )
    return any(marker in task_text for marker in choice_markers)


def _looks_like_choice_answer(answer: Any) -> bool:
    if answer is None:
        return False
    return bool(re.fullmatch(r"[A-Za-z]", str(answer).strip()))


def _looks_like_extraction_type(task_text: str, question: str) -> bool:
    extraction_markers = (
        "extract",
        "extraction",
        "entity",
        "entities",
        "concept",
        "concepts",
        "ner",
        "实体",
        "概念",
        "抽取",
        "识别",
        "板块",
    )
    if any(marker in task_text for marker in extraction_markers):
        return True
    normalized_question = _normalize_text(question)
    return any(
        marker in normalized_question
        for marker in (
            "实体识别",
            "概念板块",
            "提及的",
            "列出",
        )
    )


def _score_prediction(
    *,
    prediction: str,
    reference: str,
    question: str,
    output_type: str,
    choice_options: dict[str, str],
) -> tuple[float, Any, Any]:
    if output_type == "choice":
        return _score_choice(
            prediction=prediction,
            reference=reference,
            question=question,
            choice_options=choice_options,
        )
    if output_type == "extraction":
        return _score_extraction(prediction=prediction, reference=reference)
    normalized_prediction = _normalize_text(_extract_qa_answer(prediction))
    normalized_reference = _normalize_text(reference)
    return (
        1.0 if normalized_prediction == normalized_reference else 0.0,
        normalized_prediction,
        normalized_reference,
    )


def _score_choice(
    *,
    prediction: str,
    reference: str,
    question: str,
    choice_options: dict[str, str],
) -> tuple[float, Any, Any]:
    options = choice_options or _extract_options(question)
    prediction_norm = _extract_choice_answer(prediction, options, strict=True)
    reference_norm = _extract_choice_answer(reference, options, strict=False)
    prediction_letters = set(prediction_norm["letters"])
    reference_letters = set(reference_norm["letters"])
    if len(prediction_letters) == 1 and len(reference_letters) == 1:
        return (
            1.0 if next(iter(prediction_letters)) == next(iter(reference_letters)) else 0.0,
            prediction_norm,
            reference_norm,
        )
    prediction_option_texts = set(prediction_norm["option_texts"])
    reference_option_texts = set(reference_norm["option_texts"])
    if len(prediction_option_texts) == 1 and len(reference_option_texts) == 1:
        return (
            1.0 if next(iter(prediction_option_texts)) == next(iter(reference_option_texts)) else 0.0,
            prediction_norm,
            reference_norm,
        )
    return 0.0, prediction_norm, reference_norm


def _extract_choice_answer(value: str, options: dict[str, str], *, strict: bool) -> dict[str, Any]:
    normalized_options = {label: _normalize_text(text) for label, text in options.items()}
    text = _normalize_text(value)
    if strict:
        letters = _extract_final_choice_letters(value)
        if len(letters) == 1:
            option_texts = {normalized_options[letter] for letter in letters if letter in normalized_options}
            return {
                "letters": sorted(letters),
                "text": text,
                "option_texts": sorted(option_texts),
            }
        text_match = _extract_final_choice_text(value, normalized_options)
        if text_match is not None:
            letters = {
                label
                for label, option_text in normalized_options.items()
                if option_text == text_match
            }
            return {
                "letters": sorted(letters),
                "text": text_match,
                "option_texts": [text_match],
            }
        return {
            "letters": [],
            "text": text,
            "option_texts": [],
        }

    letters = set()
    option_texts: set[str] = set()
    exact = unicodedata.normalize("NFKC", str(value)).strip().upper()
    if re.fullmatch(r"[A-Z]", exact):
        letters.add(exact)
        if exact in normalized_options:
            option_texts.add(normalized_options[exact])
    for label, option_text in normalized_options.items():
        if text == option_text:
            letters.add(label)
            option_texts.add(option_text)
    return {
        "letters": sorted(letters),
        "text": text,
        "option_texts": sorted(option_texts),
    }


def _extract_final_choice_letters(value: str) -> set[str]:
    segments = _candidate_answer_segments(value)
    matches: list[str] = []
    for segment in segments:
        for match in FINAL_ANSWER_LETTER_RE.finditer(segment):
            matches.append(match.group(1).upper())
    if matches:
        return {matches[-1]}

    for segment in segments:
        candidate = unicodedata.normalize("NFKC", segment).strip()
        standalone = STANDALONE_CHOICE_RE.fullmatch(candidate)
        if standalone is not None:
            return {standalone.group(1).upper()}
    return set()


def _extract_final_choice_text(value: str, normalized_options: dict[str, str]) -> str | None:
    segments = _candidate_answer_segments(value)
    option_texts = set(normalized_options.values())
    for segment in segments:
        normalized_segment = _normalize_text(segment)
        if normalized_segment in option_texts:
            return normalized_segment
        stripped = re.sub(
            r"(?i)^(?:最终答案|最终选项|正确答案|正确选项|答案|选项|answer|final answer)\s*(?:is|为|是|:|：)?\s*",
            "",
            normalized_segment,
        )
        if stripped in option_texts:
            return stripped
    return None


def _candidate_answer_segments(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    segments = [segment.strip() for segment in normalized.splitlines() if segment.strip()]
    if not segments:
        return [normalized]
    return list(reversed(segments[-5:]))


def _extract_source_name(row: dict[str, Any]) -> str:
    for field_name in SOURCE_NAME_FIELDS:
        value = row.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        source_name = value.strip()
        if field_name == "source_file":
            source_name = osp.splitext(osp.basename(source_name))[0]
        return _normalize_source_name(source_name)
    return ""


def _normalize_source_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return normalized.casefold()


def _extract_row_options(row: dict[str, Any]) -> dict[str, str]:
    options: dict[str, str] = {}
    for label in OPTION_LABELS:
        value = row.get(label)
        if value is None:
            continue
        option_text = str(value).strip()
        if option_text:
            options[label] = option_text
    return options


def _coerce_choice_options(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    options: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        label = key.strip().upper()
        if label not in OPTION_LABELS:
            continue
        option_text = "" if item is None else str(item).strip()
        if option_text:
            options[label] = option_text
    return options


def _extract_options(question: str) -> dict[str, str]:
    normalized = unicodedata.normalize("NFKC", question)
    options = _extract_options_with_pattern(normalized, CHOICE_MARKER_RE)
    if not options:
        options = _extract_options_with_pattern(normalized, PAREN_CHOICE_MARKER_RE)
    return options


def _extract_options_with_pattern(question: str, pattern: re.Pattern[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for match in pattern.finditer(question):
        label = match.group("label").upper()
        text = match.group("text").strip()
        if text:
            options[label] = text
    return options


def _score_extraction(*, prediction: str, reference: str) -> tuple[float, Any, Any]:
    prediction_set = _extract_entity_set(prediction)
    reference_set = _normalize_entity_set(reference)
    return (
        1.0 if prediction_set == reference_set else 0.0,
        sorted(prediction_set),
        sorted(reference_set),
    )


def _extract_entity_set(text: str) -> set[str]:
    bold_items = [_strip_entity_candidate(item) for item in MARKDOWN_BOLD_RE.findall(text)]
    bold_items = [item for item in bold_items if item]
    if bold_items:
        return _normalize_entity_items(bold_items)

    line_items: list[str] = []
    for raw_line in str(text).splitlines():
        candidate = _strip_entity_candidate(raw_line)
        if candidate and not _looks_like_extraction_preface(candidate):
            line_items.append(candidate)
    if line_items:
        return _normalize_entity_items(line_items)

    suffix = _extract_suffix_after_delimiter(text)
    if suffix is not None:
        return _normalize_entity_set(suffix)
    return _normalize_entity_set(text)


def _strip_entity_candidate(text: str) -> str:
    candidate = unicodedata.normalize("NFKC", str(text)).strip()
    candidate = BULLET_PREFIX_RE.sub("", candidate)
    candidate = candidate.strip()
    if not candidate:
        return ""
    candidate = re.sub(r"[（(].*?$", "", candidate).strip()
    candidate = candidate.strip(" -*•\t\r\n：:")
    return candidate


def _looks_like_extraction_preface(text: str) -> bool:
    raw = unicodedata.normalize("NFKC", str(text)).strip()
    if raw.endswith(":") or raw.endswith("："):
        return True
    normalized = _normalize_text(text)
    prefaces = (
        "根据您提供",
        "根据你提供",
        "根据文本",
        "根据内容",
        "文中明确提及",
        "文中提到的",
        "提及的基金经理为",
        "提及的基金经理有",
        "提及的实体为",
        "提及的概念板块为",
        "提到的概念板块有",
        "提及的",
        "如下",
        "注",
    )
    return any(normalized.startswith(_normalize_text(prefix)) for prefix in prefaces)


def _extract_suffix_after_delimiter(text: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", str(text))
    for delimiter in ("：", ":"):
        if delimiter in normalized:
            suffix = normalized.split(delimiter)[-1].strip()
            if suffix:
                return suffix
    return None


def _normalize_entity_items(items: list[str]) -> set[str]:
    normalized_items: set[str] = set()
    for item in items:
        normalized_items.update(_normalize_entity_set(item))
    return normalized_items


def _normalize_entity_set(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = ENTITY_SEPARATORS_RE.sub(",", normalized)
    items = []
    for item in normalized.split(","):
        normalized_item = _remove_whitespace_and_punctuation(item)
        if normalized_item:
            items.append(normalized_item)
    return set(items)


def _extract_qa_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    segments = [segment.strip() for segment in normalized.splitlines() if segment.strip()]
    for segment in reversed(segments[-5:]):
        stripped = re.sub(
            r"(?i)^(?:最终答案|正确答案|答案|answer|final answer)\s*(?:is|为|是|:|：)?\s*",
            "",
            segment,
        ).strip()
        if stripped != segment:
            return stripped
    return normalized


def _remove_whitespace_and_punctuation(text: str) -> str:
    punctuation = set(string.punctuation)
    chars = []
    for char in text:
        if char.isspace():
            continue
        if char in punctuation or unicodedata.category(char).startswith("P"):
            continue
        chars.append(char)
    return "".join(chars)


def _normalize_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.strip(" \t\r\n。.!！?？,，;；:：")
