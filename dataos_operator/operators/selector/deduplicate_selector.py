"""按字段值去重的 selector。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from dataos_operator.core import Dataset, Record, SelectorOperator

KeepMode = Literal["first", "last"]


class DeduplicateSelector(SelectorOperator):
    """Exact-deduplicate records by a field value.

    Implementation: read the ``key`` field value as the dedup key (optionally
    case-insensitive) and keep the first or last duplicate per ``keep="first"/"last"``.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        key: str = "text",
        *,
        keep: KeepMode = "first",
        case_sensitive: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(key=key, keep=keep, case_sensitive=case_sensitive, **kwargs)
        if not key:
            raise ValueError("key must be a non-empty string")
        if keep not in {"first", "last"}:
            raise ValueError("keep must be one of: first, last")

        self.key = key
        self.keep = keep
        self.case_sensitive = case_sensitive

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        records = [record for dataset in data for record in dataset]
        if self.keep == "first":
            return self._keep_first(records)
        return self._keep_last(records)

    def _keep_first(self, records: Sequence[Record]) -> list[Record]:
        seen: set[str] = set()
        selected: list[Record] = []
        for record in records:
            dedup_key = self._build_dedup_key(record)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            selected.append(record)
        return selected

    def _keep_last(self, records: Sequence[Record]) -> list[Record]:
        latest_index_by_key: dict[str, int] = {}
        for index, record in enumerate(records):
            latest_index_by_key[self._build_dedup_key(record)] = index

        selected: list[Record] = []
        for index, record in enumerate(records):
            dedup_key = self._build_dedup_key(record)
            if latest_index_by_key[dedup_key] == index:
                selected.append(record)
        return selected

    def _build_dedup_key(self, record: Record) -> str:
        value = self.require_field(record, self.key)
        normalized = str(value)
        if not self.case_sensitive:
            normalized = normalized.lower()
        return normalized
