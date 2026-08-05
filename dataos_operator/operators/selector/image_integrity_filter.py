"""Filter image-text records using decoded image integrity metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator
from dataos_operator.core.image import normalize_image_metadata

_MISSING = object()


class ImageIntegrityFilter(SelectorOperator):
    """Keep records whose images exist and were decoded successfully.

    Implementation: consume the ordered metadata emitted by ``image_metadata_evaluator`` and
    require every image to have ``exists=True`` and ``readable=True``; missing and empty metadata
    are controlled independently to make pipeline behavior explicit.
    Dependency: pure-rule operator; run ``image_metadata_evaluator`` before this filter.
    """

    def __init__(
        self,
        *,
        metadata_key: str = "metadata.images",
        keep_missing: bool = False,
        keep_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            metadata_key=metadata_key,
            keep_missing=keep_missing,
            keep_empty=keep_empty,
            **kwargs,
        )
        self.metadata_key = metadata_key
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
                if not metadata:
                    if self.keep_empty:
                        selected.append(record)
                    continue
                if all(
                    item.get("exists") is True and item.get("readable") is True for item in metadata
                ):
                    selected.append(record)
        return selected
