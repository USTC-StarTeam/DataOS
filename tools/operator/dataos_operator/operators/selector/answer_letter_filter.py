"""只保留 output 里含明确字母答案的样本。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator

_MISSING = object()


def _has_letter(text: str, letters: str) -> bool:
    stripped = text.strip()
    # 整段就是一个选项字母。
    if len(stripped) == 1 and stripped.upper() in letters:
        return True
    letter_class = f"[{re.escape(letters)}]"
    patterns = (
        rf"\\boxed\{{\s*({letter_class})\s*\}}",  # \boxed{A}
        rf"####\s*({letter_class})\b",  # #### A
        # answer/option/choice cue, optional "is"/":"/"=" connector, then a letter.
        rf"(?:answer|ans|option|choice|correct)\s*(?:is|:|=|-|\.)?\s*\(?\s*({letter_class})\b",
        rf"\(\s*({letter_class})\s*\)",  # (A)
        rf"(?:^|\n)\s*({letter_class})[.):]",  # a line starting "A." / "A)" / "A:"
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


class AnswerLetterFilter(SelectorOperator):
    """Keep only rows that carry an explicit letter answer.

    Implementation: read the ``output_key`` text and keep the row when a letter from ``letters`` is
    stated as the answer — the whole field being a single letter, or a ``\\boxed{A}`` / ``#### A`` /
    ``answer is A`` / ``(A)`` / leading ``A.`` shape. The field name and letter alphabet are
    parameterized and detection is content-based (deliberately conservative to avoid matching a
    stray letter in prose), so no dataset layout is assumed. Use it to drop samples with no clear
    choice before training on a lettered multiple-choice benchmark (e.g. GPQA).
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "answer_letter_filter"

    def __init__(
        self,
        *,
        output_key: str = "output",
        letters: str = "ABCD",
        **kwargs: Any,
    ) -> None:
        super().__init__(output_key=output_key, letters=letters, **kwargs)
        self.output_key = output_key
        self.letters = letters

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                value = self.get_field(record, self.output_key, _MISSING)
                if value is _MISSING or value is None:
                    continue
                if _has_letter(str(value), self.letters):
                    selected.append(record)
        return selected
