"""把数学推理样本规范成"CoT 正文 + 统一末尾答案"的评测器格式。"""

from __future__ import annotations

import re
from typing import Any

from dataos_operator.core import MapperOperator, Record

_MISSING = object()

#: 依次尝试的默认答案抽取正则:``\boxed{...}`` -> ``#### x`` -> 文本末尾数字。
DEFAULT_ANSWER_PATTERNS: tuple[str, ...] = (
    r"\\boxed\{([^{}]*)\}",
    r"####\s*([^\n]+)",
    r"(-?\d[\d,]*\.?\d*)\s*$",
)
_TRAILING_MARKER = re.compile(r"\s*####\s*[^\n]*\s*$")


def extract_answer(text: str, pattern: str | None = None) -> str | None:
    """基于内容抽取最终答案:给了 ``pattern`` 就用它,否则跑默认级联正则。"""

    if pattern is not None:
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            return None
        captured = match.group(1) if match.groups() else match.group(0)
        return captured.strip() or None
    for candidate in DEFAULT_ANSWER_PATTERNS:
        match = re.search(candidate, text, flags=re.DOTALL)
        if match:
            captured = match.group(1) if match.groups() else match.group(0)
            captured = captured.strip()
            if captured:
                return captured
    return None


class CotAnswerFormatter(MapperOperator):
    """Format math reasoning samples as CoT body plus a unified trailing answer line.

    Implementation: read the ``reasoning_key`` chain-of-thought text and the answer — taken from
    ``answer_key`` when provided, otherwise extracted from the reasoning via ``answer_pattern``
    (defaulting to a ``\\boxed{}`` / ``#### x`` / trailing-number cascade). Any pre-existing
    ``#### ...`` marker is stripped off the reasoning body and replaced by ``answer_template``
    formatted with the resolved answer, so gsm8k-style and aime/``\\boxed`` sources collapse to
    one extractable format. Field names are parameterized and answer detection is content-based;
    records without a resolvable answer are left with the reasoning body unchanged.
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "cot_answer_formatter"

    def __init__(
        self,
        question_key: str = "question",
        reasoning_key: str = "reasoning",
        answer_key: str | None = None,
        *,
        output_key: str = "output",
        answer_pattern: str | None = None,
        answer_template: str = "#### {answer}",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            question_key=question_key,
            reasoning_key=reasoning_key,
            answer_key=answer_key,
            output_key=output_key,
            answer_pattern=answer_pattern,
            answer_template=answer_template,
            **kwargs,
        )
        self.question_key = question_key
        self.reasoning_key = reasoning_key
        self.answer_key = answer_key
        self.output_key = output_key
        self.answer_pattern = answer_pattern
        self.answer_template = answer_template

    def map_record(self, record: Record) -> Record:
        reasoning = _as_text(self.get_field(record, self.reasoning_key, ""))
        answer = self._resolve_answer(record, reasoning)

        body = _TRAILING_MARKER.sub("", reasoning).rstrip()
        if answer is not None:
            suffix = self.answer_template.format(answer=answer)
            output = f"{body}\n{suffix}" if body else suffix
        else:
            output = body

        result = dict(record)
        result[self.output_key] = output
        if answer is not None and self.answer_key:
            result[self.answer_key] = answer
        return result

    def _resolve_answer(self, record: Record, reasoning: str) -> str | None:
        if self.answer_key:
            explicit = self.get_field(record, self.answer_key, _MISSING)
            if explicit is not _MISSING and explicit is not None and str(explicit).strip():
                return str(explicit).strip()
        return extract_answer(reasoning, self.answer_pattern)


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)
