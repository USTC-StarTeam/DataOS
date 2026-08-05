"""Filter records by decoded image aspect ratio."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator
from dataos_operator.core.image import image_strategy_matches, normalize_image_metadata

_MISSING = object()


class ImageAspectRatioFilter(SelectorOperator):
    """Keep records whose image width-to-height ratios fall within inclusive bounds.

    Implementation: read ``aspect_ratio`` metadata for each image, enforce the configured minimum
    and maximum, and combine multi-image decisions with ``strategy="all"`` or ``strategy="any"``.
    Dependency: pure-rule operator; run ``image_metadata_evaluator`` before this filter.
    """

    def __init__(
        self,
        *,
        metadata_key: str = "metadata.images",
        min_ratio: float | None = 0.0,
        max_ratio: float | None = None,
        strategy: str = "all",
        keep_missing: bool = False,
        keep_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            metadata_key=metadata_key,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            strategy=strategy,
            keep_missing=keep_missing,
            keep_empty=keep_empty,
            **kwargs,
        )
        if min_ratio is not None and min_ratio < 0:
            raise ValueError("min_ratio must be >= 0")
        if max_ratio is not None and max_ratio < 0:
            raise ValueError("max_ratio must be >= 0")
        if min_ratio is not None and max_ratio is not None and min_ratio > max_ratio:
            raise ValueError("min_ratio must be <= max_ratio")
        if strategy not in {"any", "all"}:
            raise ValueError("strategy must be either 'any' or 'all'")
        self.metadata_key = metadata_key
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
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
        value = item.get("aspect_ratio")
        if not isinstance(value, int | float) or isinstance(value, bool):
            return False
        ratio = float(value)
        if self.min_ratio is not None and ratio < self.min_ratio:
            return False
        return self.max_ratio is None or ratio <= self.max_ratio
