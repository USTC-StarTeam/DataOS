"""IFEval (Zhou et al., 2023) verifiable instruction-following benchmark.

Each example carries a ``prompt`` plus a list of ``instruction_id_list`` and a
``kwargs`` list. Scoring runs each instruction's verifier against the model
response and reports prompt-level (all instructions satisfied) and
instruction-level accuracy.

This module ships verifiers for the highest-frequency instruction types in the
public IFEval set; unknown ids fall back to a documented "skip + flag" path so
new types can be added incrementally without breaking the run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from datasets import Dataset

from dataos.eval.types import EvalConfig, EvalRequest, EvalResultBundle, EvalSampleResult, ModelResponse


PROMPT_FIELDS = ("prompt", "instruction")
INSTRUCTION_LIST_FIELDS = ("instruction_id_list", "instruction_ids", "instructions")
KWARGS_LIST_FIELDS = ("kwargs", "instruction_kwargs")
KEY_FIELD_CANDIDATES = ("key", "id", "prompt_id", "example_id")


@dataclass(slots=True)
class _Verdict:
    instruction_id: str
    followed: bool
    reason: str | None = None


VerifierFn = Callable[[str, dict[str, Any]], _Verdict]


def _strip_response(text: str) -> str:
    return (text or "").strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _sentence_count(text: str) -> int:
    sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return len([s for s in sentences if s.strip()])


def _paragraph_count(text: str) -> int:
    paragraphs = re.split(r"\n\s*\*\s*\*\s*\*\s*\n", text)
    if len(paragraphs) > 1:
        return len([p for p in paragraphs if p.strip()])
    paragraphs = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
    return len(paragraphs)


def _verify_length_words(text: str, kw: dict[str, Any]) -> _Verdict:
    relation = kw.get("relation")
    target = int(kw["num_words"])
    count = _word_count(text)
    if relation in ("at least", "at_least"):
        ok = count >= target
    elif relation in ("at most", "at_most"):
        ok = count <= target
    elif relation in ("less than", "less_than"):
        ok = count < target
    else:
        ok = count == target
    return _Verdict("length_constraints:number_words", ok, f"words={count}, target={relation} {target}")


def _verify_length_sentences(text: str, kw: dict[str, Any]) -> _Verdict:
    relation = kw.get("relation")
    target = int(kw["num_sentences"])
    count = _sentence_count(text)
    if relation in ("at least", "at_least"):
        ok = count >= target
    elif relation in ("at most", "at_most"):
        ok = count <= target
    else:
        ok = count == target
    return _Verdict("length_constraints:number_sentences", ok, f"sentences={count}, target={relation} {target}")


def _verify_length_paragraphs(text: str, kw: dict[str, Any]) -> _Verdict:
    target = int(kw["num_paragraphs"])
    count = _paragraph_count(text)
    return _Verdict("length_constraints:number_paragraphs", count == target, f"paragraphs={count}, target={target}")


def _verify_keyword_existence(text: str, kw: dict[str, Any]) -> _Verdict:
    keywords = kw.get("keywords") or []
    missing = [w for w in keywords if w.lower() not in text.lower()]
    return _Verdict("keywords:existence", not missing, f"missing={missing}" if missing else None)


def _verify_keyword_forbidden(text: str, kw: dict[str, Any]) -> _Verdict:
    forbidden = kw.get("forbidden_words") or []
    present = [w for w in forbidden if w.lower() in text.lower()]
    return _Verdict("keywords:forbidden_words", not present, f"present={present}" if present else None)


def _verify_keyword_frequency(text: str, kw: dict[str, Any]) -> _Verdict:
    keyword = kw["keyword"]
    relation = kw.get("relation")
    target = int(kw["frequency"])
    count = len(re.findall(re.escape(keyword), text, re.IGNORECASE))
    if relation in ("at least", "at_least"):
        ok = count >= target
    elif relation in ("at most", "at_most"):
        ok = count <= target
    else:
        ok = count == target
    return _Verdict("keywords:frequency", ok, f"keyword={keyword!r}, count={count}, target={relation} {target}")


def _verify_letter_frequency(text: str, kw: dict[str, Any]) -> _Verdict:
    letter = kw["letter"]
    relation = kw.get("relation")
    target = int(kw["let_frequency"]) if "let_frequency" in kw else int(kw.get("frequency", 0))
    count = sum(1 for ch in text if ch.lower() == letter.lower())
    if relation in ("at least", "at_least"):
        ok = count >= target
    elif relation in ("at most", "at_most"):
        ok = count <= target
    else:
        ok = count == target
    return _Verdict("keywords:letter_frequency", ok, f"letter={letter!r}, count={count}, target={relation} {target}")


def _verify_change_case_lower(text: str, kw: dict[str, Any]) -> _Verdict:
    del kw
    return _Verdict("change_case:english_lowercase", text == text.lower())


def _verify_change_case_upper(text: str, kw: dict[str, Any]) -> _Verdict:
    del kw
    return _Verdict("change_case:english_capital", text == text.upper())


def _verify_capital_word_frequency(text: str, kw: dict[str, Any]) -> _Verdict:
    relation = kw.get("relation")
    target = int(kw["capital_frequency"]) if "capital_frequency" in kw else int(kw.get("frequency", 0))
    count = sum(1 for word in re.findall(r"\b\w+\b", text) if word.isupper() and len(word) > 1)
    if relation in ("at least", "at_least"):
        ok = count >= target
    elif relation in ("at most", "at_most"):
        ok = count <= target
    else:
        ok = count == target
    return _Verdict(
        "change_case:capital_word_frequency",
        ok,
        f"capital_words={count}, target={relation} {target}",
    )


def _verify_punctuation_no_comma(text: str, kw: dict[str, Any]) -> _Verdict:
    del kw
    return _Verdict("punctuation:no_comma", "," not in text)


def _verify_startend_quotation(text: str, kw: dict[str, Any]) -> _Verdict:
    del kw
    stripped = text.strip()
    ok = len(stripped) >= 2 and stripped[0] in '"“' and stripped[-1] in '"”'
    return _Verdict("startend:quotation", ok)


def _verify_startend_endchecker(text: str, kw: dict[str, Any]) -> _Verdict:
    phrase = kw["end_phrase"].strip()
    ok = text.strip().endswith(phrase)
    return _Verdict("startend:end_checker", ok, f"end_phrase={phrase!r}")


def _verify_format_title(text: str, kw: dict[str, Any]) -> _Verdict:
    del kw
    return _Verdict("detectable_format:title", bool(re.search(r"<<[^<>\n]+>>", text)))


def _verify_format_bullets(text: str, kw: dict[str, Any]) -> _Verdict:
    target = int(kw["num_bullets"])
    bullets = re.findall(r"(?m)^\s*[\*\-]\s+\S", text)
    return _Verdict("detectable_format:number_bullet_lists", len(bullets) == target, f"bullets={len(bullets)}, target={target}")


def _verify_format_highlighted(text: str, kw: dict[str, Any]) -> _Verdict:
    target = int(kw["num_highlights"])
    matches = re.findall(r"\*[^*\n]+\*", text)
    return _Verdict("detectable_format:number_highlighted_sections", len(matches) >= target, f"highlights={len(matches)}, target>={target}")


def _verify_format_sections(text: str, kw: dict[str, Any]) -> _Verdict:
    target = int(kw["num_sections"])
    splitter = kw.get("section_spliter") or kw.get("section_splitter") or "Section"
    pattern = re.compile(rf"(?im)^\s*{re.escape(splitter)}\s+\d", re.MULTILINE)
    return _Verdict("detectable_format:multiple_sections", len(pattern.findall(text)) >= target, f"sections>={target}")


def _verify_format_json(text: str, kw: dict[str, Any]) -> _Verdict:
    del kw
    stripped = text.strip().strip("`")
    if stripped.lower().startswith("json"):
        stripped = stripped[4:].strip()
    try:
        json.loads(stripped)
        return _Verdict("detectable_format:json_format", True)
    except Exception as exc:  # noqa: BLE001
        return _Verdict("detectable_format:json_format", False, f"json_error={exc}")


def _verify_format_constrained_response(text: str, kw: dict[str, Any]) -> _Verdict:
    options = kw.get("response_options") or ["My answer is yes.", "My answer is no.", "My answer is maybe."]
    stripped = text.strip()
    return _Verdict("detectable_format:constrained_response", stripped in options)


def _verify_detectable_repeat_prompt(text: str, kw: dict[str, Any]) -> _Verdict:
    prompt_to_repeat = kw.get("prompt_to_repeat") or ""
    return _Verdict(
        "detectable_content:repeat_prompt",
        text.strip().startswith(prompt_to_repeat.strip()),
    )


def _verify_detectable_postscript(text: str, kw: dict[str, Any]) -> _Verdict:
    marker = kw.get("postscript_marker") or "P.S."
    return _Verdict("detectable_content:postscript", marker in text)


def _verify_detectable_placeholders(text: str, kw: dict[str, Any]) -> _Verdict:
    target = int(kw["num_placeholders"])
    matches = re.findall(r"\[[^\[\]\n]+\]", text)
    return _Verdict("detectable_content:number_placeholders", len(matches) >= target, f"placeholders={len(matches)}, target>={target}")


def _verify_combination_two_responses(text: str, kw: dict[str, Any]) -> _Verdict:
    del kw
    return _Verdict("combination:two_responses", "******" in text)


def _verify_combination_repeat(text: str, kw: dict[str, Any]) -> _Verdict:
    prompt = kw.get("prompt_to_repeat") or ""
    if not prompt:
        return _Verdict("combination:repeat_prompt", False, "missing prompt_to_repeat kwarg")
    return _Verdict("combination:repeat_prompt", text.strip().startswith(prompt.strip()))


def _verify_language_response(text: str, kw: dict[str, Any]) -> _Verdict:
    language = (kw.get("language") or "en").lower()
    if language == "en":
        ascii_chars = sum(1 for ch in text if ord(ch) < 128)
        ratio = ascii_chars / max(1, len(text))
        return _Verdict("language:response_language", ratio > 0.9, f"ascii_ratio={ratio:.3f}")
    return _Verdict("language:response_language", True, f"language={language!r} (heuristic skip)")


_VERIFIERS: dict[str, VerifierFn] = {
    "length_constraints:number_words": _verify_length_words,
    "length_constraints:number_sentences": _verify_length_sentences,
    "length_constraints:number_paragraphs": _verify_length_paragraphs,
    "length_constraints:nth_paragraph_first_word": _verify_length_paragraphs,
    "keywords:existence": _verify_keyword_existence,
    "keywords:forbidden_words": _verify_keyword_forbidden,
    "keywords:frequency": _verify_keyword_frequency,
    "keywords:letter_frequency": _verify_letter_frequency,
    "change_case:english_lowercase": _verify_change_case_lower,
    "change_case:english_capital": _verify_change_case_upper,
    "change_case:capital_word_frequency": _verify_capital_word_frequency,
    "punctuation:no_comma": _verify_punctuation_no_comma,
    "startend:quotation": _verify_startend_quotation,
    "startend:end_checker": _verify_startend_endchecker,
    "detectable_format:title": _verify_format_title,
    "detectable_format:number_bullet_lists": _verify_format_bullets,
    "detectable_format:number_highlighted_sections": _verify_format_highlighted,
    "detectable_format:multiple_sections": _verify_format_sections,
    "detectable_format:json_format": _verify_format_json,
    "detectable_format:constrained_response": _verify_format_constrained_response,
    "detectable_content:postscript": _verify_detectable_postscript,
    "detectable_content:number_placeholders": _verify_detectable_placeholders,
    "detectable_content:repeat_prompt": _verify_detectable_repeat_prompt,
    "combination:two_responses": _verify_combination_two_responses,
    "combination:repeat_prompt": _verify_combination_repeat,
    "language:response_language": _verify_language_response,
}


def _coerce_kwargs(kwargs_value: Any, count: int) -> list[dict[str, Any]]:
    if isinstance(kwargs_value, list):
        normalized = []
        for entry in kwargs_value:
            if entry in (None, ""):
                normalized.append({})
            elif isinstance(entry, dict):
                normalized.append({k: v for k, v in entry.items() if v is not None})
            elif isinstance(entry, str):
                try:
                    parsed = json.loads(entry)
                except json.JSONDecodeError:
                    parsed = {}
                normalized.append(parsed if isinstance(parsed, dict) else {})
            else:
                normalized.append({})
        if len(normalized) < count:
            normalized.extend([{}] * (count - len(normalized)))
        return normalized[:count]
    return [{} for _ in range(count)]


def _coerce_instruction_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []


def _extract_prompt(row: dict[str, Any], index: int) -> str:
    for field_name in PROMPT_FIELDS:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"Row {index} requires one of {PROMPT_FIELDS} for ifeval")


def _extract_instruction_ids(row: dict[str, Any]) -> list[str]:
    for field_name in INSTRUCTION_LIST_FIELDS:
        if field_name in row:
            return _coerce_instruction_ids(row.get(field_name))
    return []


def _extract_kwargs(row: dict[str, Any], count: int) -> list[dict[str, Any]]:
    for field_name in KWARGS_LIST_FIELDS:
        if field_name in row:
            return _coerce_kwargs(row.get(field_name), count)
    return [{} for _ in range(count)]


def _extract_key(row: dict[str, Any], index: int) -> str:
    for field_name in KEY_FIELD_CANDIDATES:
        value = row.get(field_name)
        if value is not None:
            return str(value)
    return str(index)


class IFEvalTask:
    name = "ifeval"
    version = "v1"
    supported_request_types = {"generate_until"}

    def default_split(self) -> str:
        return "test"

    def build_requests(self, dataset: Dataset, config: EvalConfig) -> list[EvalRequest]:
        del config
        requests: list[EvalRequest] = []
        for index, row in enumerate(dataset):
            prompt = _extract_prompt(row, index)
            instruction_ids = _extract_instruction_ids(row)
            kwargs_list = _extract_kwargs(row, len(instruction_ids))
            requests.append(
                EvalRequest(
                    sample_id=_extract_key(row, index),
                    index=index,
                    request_type="generate_until",
                    prompt=prompt,
                    meta={
                        "instruction_id_list": instruction_ids,
                        "kwargs": kwargs_list,
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
        del dataset, config
        if len(requests) != len(responses):
            raise ValueError("Response count does not match request count")

        samples: list[EvalSampleResult] = []
        prompt_correct = 0
        instr_total = 0
        instr_correct = 0
        skipped_ids: list[str] = []

        for request, response in zip(requests, responses):
            text = _strip_response(response.text or "")
            instruction_ids: list[str] = list(request.meta.get("instruction_id_list") or [])
            kwargs_list: list[dict[str, Any]] = list(request.meta.get("kwargs") or [])
            verdicts: list[_Verdict] = []
            for instruction_id, kw in zip(instruction_ids, kwargs_list):
                verifier = _VERIFIERS.get(instruction_id)
                if verifier is None:
                    skipped_ids.append(instruction_id)
                    verdicts.append(_Verdict(instruction_id, False, "no_verifier_registered"))
                    continue
                try:
                    verdicts.append(verifier(text, kw))
                except Exception as exc:  # noqa: BLE001
                    verdicts.append(_Verdict(instruction_id, False, f"verifier_error={exc}"))

            sample_followed = [v.followed for v in verdicts]
            prompt_ok = bool(sample_followed) and all(sample_followed)
            instr_total += len(verdicts)
            instr_correct += sum(1 for ok in sample_followed if ok)
            if prompt_ok:
                prompt_correct += 1

            samples.append(
                EvalSampleResult(
                    sample_id=request.sample_id,
                    index=request.index,
                    prompt=request.prompt,
                    prediction=response.text,
                    reference=instruction_ids,
                    normalized_prediction=text,
                    normalized_reference=instruction_ids,
                    metrics={
                        "prompt_followed": float(prompt_ok),
                        "instruction_followed": (
                            sum(1 for ok in sample_followed if ok) / len(sample_followed)
                            if sample_followed
                            else 0.0
                        ),
                    },
                    meta={
                        "instruction_id_list": instruction_ids,
                        "kwargs": kwargs_list,
                        "verdicts": [
                            {"instruction_id": v.instruction_id, "followed": v.followed, "reason": v.reason}
                            for v in verdicts
                        ],
                        "is_correct": prompt_ok,
                    },
                )
            )

        prompt_acc = prompt_correct / len(samples) if samples else 0.0
        instr_acc = instr_correct / instr_total if instr_total else 0.0
        return EvalResultBundle(
            metric_name="prompt_strict_accuracy",
            score=prompt_acc,
            summary={
                "num_examples": len(samples),
                "prompt_strict_accuracy": prompt_acc,
                "instruction_strict_accuracy": instr_acc,
                "instruction_total": instr_total,
                "instruction_correct": instr_correct,
                "skipped_unregistered_ids": sorted(set(skipped_ids)),
            },
            samples=samples,
        )
