"""Filter records by encoded image file size."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator
from dataos_operator.core.image import image_strategy_matches, normalize_image_metadata

_MISSING = object()


class ImageSizeFilter(SelectorOperator):
    """Keep records whose encoded image byte sizes satisfy inclusive bounds.

    Implementation: read the on-disk ``bytes`` value for every image and combine its range checks
    with ``strategy="all"`` or ``strategy="any"``; missing and empty metadata handling is explicit.
    Dependency: pure-rule operator; run ``image_metadata_evaluator`` before this filter.
    """

    def __init__(
        self,
        *,
        metadata_key: str = "metadata.images",
        min_bytes: int | None = 1,
        max_bytes: int | None = None,
        strategy: str = "all",
        keep_missing: bool = False,
        keep_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            metadata_key=metadata_key,
            min_bytes=min_bytes,
            max_bytes=max_bytes,
            strategy=strategy,
            keep_missing=keep_missing,
            keep_empty=keep_empty,
            **kwargs,
        )
        if min_bytes is not None and min_bytes < 0:
            raise ValueError("min_bytes must be >= 0")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        if min_bytes is not None and max_bytes is not None and min_bytes > max_bytes:
            raise ValueError("min_bytes must be <= max_bytes")
        if strategy not in {"any", "all"}:
            raise ValueError("strategy must be either 'any' or 'all'")
        self.metadata_key = metadata_key
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.strategy = strategy
        self.keep_missing = keep_missing
        self.keep_empty = keep_empty

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                value = self.get_field(record, self.metadata_key, _MISSING)
                if value is _MISSING:
                    if self.keep_missing:
                        selected.append(record)
                    continue
                metadata = normalize_image_metadata(value, strict=False)
                results = [self._matches(item) for item in metadata]
                if image_strategy_matches(
                    results,
                    strategy=self.strategy,
                    keep_empty=self.keep_empty,
                ):
                    selected.append(record)
        return selected

    def _matches(self, item: Mapping[str, Any]) -> bool:
        value = item.get("bytes")
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if self.min_bytes is not None and value < self.min_bytes:
            return False
        return self.max_bytes is None or value <= self.max_bytes
