"""只保留能从 output 中抽取到最终答案的样本。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator

_MISSING = object()

#: 依次尝试的默认答案抽取正则:``\boxed{...}`` -> ``#### x`` -> 文本末尾数字。
DEFAULT_ANSWER_PATTERNS: tuple[str, ...] = (
    r"\\boxed\{([^{}]*)\}",
    r"####\s*([^\n]+)",
    r"(-?\d[\d,]*\.?\d*)\s*$",
)


def _has_answer(text: str, pattern: str | None) -> bool:
    candidates = (pattern,) if pattern is not None else DEFAULT_ANSWER_PATTERNS
    for candidate in candidates:
        match = re.search(candidate, text, flags=re.DOTALL)
        if match:
            captured = match.group(1) if match.groups() else match.group(0)
            if captured.strip():
                return True
    return False


class HasAnswerFilter(SelectorOperator):
    """Keep only rows from which a final answer can be extracted.

    Implementation: read the ``output_key`` text and keep the row when ``answer_pattern`` matches
    (defaulting to a ``\\boxed{}`` / ``#### x`` / trailing-number cascade, so both aime-style and
    gsm8k-style sources are covered). The field name and pattern are parameterized and the check is
    content-based, so no dataset layout is assumed. Use it to drop samples whose target has no
    recoverable answer before training on a math benchmark that scores the extracted answer.
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "has_answer_filter"

    def __init__(
        self,
        *,
        output_key: str = "output",
        answer_pattern: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(output_key=output_key, answer_pattern=answer_pattern, **kwargs)
        self.output_key = output_key
        self.answer_pattern = answer_pattern

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                value = self.get_field(record, self.output_key, _MISSING)
                if value is _MISSING or value is None:
                    continue
                if _has_answer(str(value), self.answer_pattern):
                    selected.append(record)
        return selected
