"""基于长度区间的 selector 示例。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator

_MISSING = object()


class LengthFilter(SelectorOperator):
    """Filter samples by text length, keeping records whose length falls within the range.

    Implementation: read the ``text_key`` text (optionally stripped of whitespace) and keep
    it when its character count lies in ``[min_length, max_length]``; missing fields are kept
    or dropped per ``keep_missing``.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        *,
        keep_missing: bool = False,
        strip_whitespace: bool = False,
        length_unit: str = "char",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            min_length=min_length,
            max_length=max_length,
            keep_missing=keep_missing,
            strip_whitespace=strip_whitespace,
            length_unit=length_unit,
            **kwargs,
        )
        if min_length is None and max_length is None:
            raise ValueError("At least one of min_length or max_length must be set")
        if min_length is not None and min_length < 0:
            raise ValueError("min_length must be >= 0")
        if max_length is not None and max_length < 0:
            raise ValueError("max_length must be >= 0")
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError("min_length must be <= max_length")
        if length_unit != "char":
            raise ValueError(f"Unsupported length_unit: {length_unit!r}")

        self.min_length = min_length
        self.max_length = max_length
        self.keep_missing = keep_missing
        self.strip_whitespace = strip_whitespace
        self.length_unit = length_unit

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                maybe_selected = self._select_record(record)
                if maybe_selected is not None:
                    selected.append(maybe_selected)
        return selected

    def _select_record(self, record: Record) -> Record | None:
        text = self.get_field(record, self.text_key, _MISSING)
        if text is _MISSING:
            return record if self.keep_missing else None

        normalized_text = str(text)
        if self.strip_whitespace:
            normalized_text = normalized_text.strip()
        length = len(normalized_text)
        if self.min_length is not None and length < self.min_length:
            return None
        if self.max_length is not None and length > self.max_length:
            return None
        return record
