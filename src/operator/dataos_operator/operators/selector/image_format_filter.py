"""Filter records by decoded image format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator
from dataos_operator.core.image import image_strategy_matches, normalize_image_metadata

_DEFAULT_FORMATS = ("JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF")
_FORMAT_ALIASES = {"JPG": "JPEG", "TIF": "TIFF"}
_MISSING = object()


class ImageFormatFilter(SelectorOperator):
    """Keep records whose decoded image formats belong to an allowlist.

    Implementation: normalize format names case-insensitively (including JPG/JPEG and TIF/TIFF
    aliases), test each metadata item against ``allowed_formats``, and combine multi-image results
    with ``strategy="all"`` or ``strategy="any"``.
    Dependency: pure-rule operator; run ``image_metadata_evaluator`` before this filter.
    """

    def __init__(
        self,
        *,
        allowed_formats: Sequence[str] = _DEFAULT_FORMATS,
        metadata_key: str = "metadata.images",
        strategy: str = "all",
        keep_missing: bool = False,
        keep_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(allowed_formats, str | bytes | bytearray):
            raise ValueError("allowed_formats must be a sequence of format names")
        normalized_formats = tuple(_normalize_format(value) for value in allowed_formats)
        if not normalized_formats:
            raise ValueError("allowed_formats must not be empty")
        super().__init__(
            allowed_formats=list(normalized_formats),
            metadata_key=metadata_key,
            strategy=strategy,
            keep_missing=keep_missing,
            keep_empty=keep_empty,
            **kwargs,
        )
        if strategy not in {"any", "all"}:
            raise ValueError("strategy must be either 'any' or 'all'")
        self.allowed_formats = frozenset(normalized_formats)
        self.metadata_key = metadata_key
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
        value = item.get("format")
        return isinstance(value, str) and _normalize_format(value) in self.allowed_formats


def _normalize_format(value: str) -> str:
    normalized = str(value).strip().upper()
    if not normalized:
        raise ValueError("image format names must not be empty")
    return _FORMAT_ALIASES.get(normalized, normalized)
