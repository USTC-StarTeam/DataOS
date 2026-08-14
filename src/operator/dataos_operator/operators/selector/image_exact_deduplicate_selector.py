"""Exact-deduplicate image records using precomputed content hashes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from dataos_operator.core import Record, SelectorOperator
from dataos_operator.core.image import normalize_image_metadata

KeepMode = Literal["first", "last"]


class ImageExactDeduplicateSelector(SelectorOperator):
    """Remove records that reference byte-identical image sets.

    Implementation: build a sample key from ordered hashes in ``metadata_key`` and keep either the
    first or last occurrence; set ``order_sensitive=False`` when image order should not distinguish
    otherwise identical multi-image samples.
    Dependency: pure-rule operator; run ``image_hash_evaluator`` before this selector.
    """

    def __init__(
        self,
        *,
        metadata_key: str = "metadata.images",
        hash_key: str = "sha256",
        keep: KeepMode = "first",
        order_sensitive: bool = True,
        keep_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            metadata_key=metadata_key,
            hash_key=hash_key,
            keep=keep,
            order_sensitive=order_sensitive,
            keep_missing=keep_missing,
            **kwargs,
        )
        if not hash_key:
            raise ValueError("hash_key must not be empty")
        if keep not in {"first", "last"}:
            raise ValueError("keep must be one of: first, last")
        self.metadata_key = metadata_key
        self.hash_key = hash_key
        self.keep = keep
        self.order_sensitive = order_sensitive
        self.keep_missing = keep_missing

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        records = [record for dataset in data for record in dataset]
        keys = [self._build_key(record) for record in records]
        if self.keep == "first":
            return self._keep_first(records, keys)
        return self._keep_last(records, keys)

    def _build_key(self, record: Record) -> tuple[str, ...] | None:
        metadata = normalize_image_metadata(
            self.get_field(record, self.metadata_key, None),
            strict=False,
        )
        hashes: list[str] = []
        for item in metadata:
            value = item.get(self.hash_key)
            if not isinstance(value, str) or not value:
                return None
            hashes.append(value)
        if not hashes:
            return None
        if not self.order_sensitive:
            hashes.sort()
        return tuple(hashes)

    def _keep_first(
        self,
        records: Sequence[Record],
        keys: Sequence[tuple[str, ...] | None],
    ) -> list[Record]:
        seen: set[tuple[str, ...]] = set()
        selected: list[Record] = []
        for record, key in zip(records, keys, strict=True):
            if key is None:
                if self.keep_missing:
                    selected.append(record)
                continue
            if key not in seen:
                seen.add(key)
                selected.append(record)
        return selected

    def _keep_last(
        self,
        records: Sequence[Record],
        keys: Sequence[tuple[str, ...] | None],
    ) -> list[Record]:
        latest: dict[tuple[str, ...], int] = {}
        for index, key in enumerate(keys):
            if key is not None:
                latest[key] = index
        selected: list[Record] = []
        for index, (record, key) in enumerate(zip(records, keys, strict=True)):
            if key is None:
                if self.keep_missing:
                    selected.append(record)
            elif latest[key] == index:
                selected.append(record)
        return selected
