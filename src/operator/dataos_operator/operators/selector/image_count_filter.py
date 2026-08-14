"""Filter records by the number of referenced images."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator
from dataos_operator.core.image import normalize_image_list

_MISSING = object()


class ImageCountFilter(SelectorOperator):
    """Keep records whose top-level image count falls within an inclusive range.

    Implementation: normalize ``images_key`` to an ordered list, then retain samples with a count
    in ``[min_count, max_count]``; a genuinely absent field follows ``keep_missing`` instead of
    being silently treated as an empty image list.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        min_count: int = 1,
        max_count: int | None = None,
        *,
        images_key: str = "images",
        keep_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            min_count=min_count,
            max_count=max_count,
            images_key=images_key,
            keep_missing=keep_missing,
            **kwargs,
        )
        if min_count < 0:
            raise ValueError("min_count must be >= 0")
        if max_count is not None and max_count < min_count:
            raise ValueError("max_count must be >= min_count")
        self.min_count = min_count
        self.max_count = max_count
        self.images_key = images_key
        self.keep_missing = keep_missing

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                value = self.get_field(record, self.images_key, _MISSING)
                if value is _MISSING:
                    if self.keep_missing:
                        selected.append(record)
                    continue
                count = len(normalize_image_list(value, strict=False))
                if count < self.min_count:
                    continue
                if self.max_count is not None and count > self.max_count:
                    continue
                selected.append(record)
        return selected
