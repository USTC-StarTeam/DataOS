"""Filter records by decoded image dimensions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator
from dataos_operator.core.image import image_strategy_matches, normalize_image_metadata

_MISSING = object()


class ImageShapeFilter(SelectorOperator):
    """Keep records whose decoded image dimensions satisfy configurable bounds.

    Implementation: test width, height, and pixel-count metadata for each image, then combine the
    per-image decisions with ``strategy="all"`` or ``strategy="any"``; bounds are inclusive.
    Dependency: pure-rule operator; run ``image_metadata_evaluator`` before this filter.
    """

    def __init__(
        self,
        *,
        metadata_key: str = "metadata.images",
        min_width: int | None = 1,
        max_width: int | None = None,
        min_height: int | None = 1,
        max_height: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        strategy: str = "all",
        keep_missing: bool = False,
        keep_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        bounds = {
            "min_width": min_width,
            "max_width": max_width,
            "min_height": min_height,
            "max_height": max_height,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
        super().__init__(
            metadata_key=metadata_key,
            **bounds,
            strategy=strategy,
            keep_missing=keep_missing,
            keep_empty=keep_empty,
            **kwargs,
        )
        _validate_integer_bounds(bounds)
        if min_width is not None and max_width is not None and min_width > max_width:
            raise ValueError("min_width must be <= max_width")
        if min_height is not None and max_height is not None and min_height > max_height:
            raise ValueError("min_height must be <= max_height")
        if min_pixels is not None and max_pixels is not None and min_pixels > max_pixels:
            raise ValueError("min_pixels must be <= max_pixels")
        if strategy not in {"any", "all"}:
            raise ValueError("strategy must be either 'any' or 'all'")
        self.metadata_key = metadata_key
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
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
        return (
            _value_in_range(item.get("width"), self.min_width, self.max_width)
            and _value_in_range(item.get("height"), self.min_height, self.max_height)
            and _value_in_range(item.get("pixels"), self.min_pixels, self.max_pixels)
        )


def _validate_integer_bounds(bounds: Mapping[str, int | None]) -> None:
    for name, value in bounds.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be >= 0")


def _value_in_range(value: Any, minimum: int | None, maximum: int | None) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if minimum is not None and value < minimum:
        return False
    return maximum is None or value <= maximum
