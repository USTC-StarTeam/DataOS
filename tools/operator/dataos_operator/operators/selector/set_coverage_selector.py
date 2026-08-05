"""按集合覆盖率与优先级选择样本的通用 selector。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from dataos_operator.core import Dataset, Record, SelectorOperator


class SetCoverageSelector(SelectorOperator):
    """Greedily select samples by the new coverage a set field adds, maximizing diversity.

    Implementation: optionally sort by ``priority_key`` first, then scan greedily—a record is
    selected (and its items accumulated into coverage) only when it adds ``min_new_items`` or
    more new set elements; stop at ``target_size`` or ``max_coverage_ratio``;
    ``fill_remaining`` backfills from unselected records.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        set_key: str,
        *,
        priority_key: str | None = None,
        target_size: int | None = None,
        min_new_items: int = 1,
        max_coverage_ratio: float | None = None,
        keep_missing: bool = False,
        reverse: bool = True,
        fill_remaining: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            set_key=set_key,
            priority_key=priority_key,
            target_size=target_size,
            min_new_items=min_new_items,
            max_coverage_ratio=max_coverage_ratio,
            keep_missing=keep_missing,
            reverse=reverse,
            fill_remaining=fill_remaining,
            **kwargs,
        )
        if not set_key:
            raise ValueError("set_key must be a non-empty string")
        if target_size is not None and target_size <= 0:
            raise ValueError("target_size must be greater than zero")
        if min_new_items <= 0:
            raise ValueError("min_new_items must be greater than zero")
        if max_coverage_ratio is not None and not 0.0 < max_coverage_ratio <= 1.0:
            raise ValueError("max_coverage_ratio must be in (0, 1]")

        self.set_key = set_key
        self.priority_key = priority_key
        self.target_size = target_size
        self.min_new_items = min_new_items
        self.max_coverage_ratio = max_coverage_ratio
        self.keep_missing = keep_missing
        self.reverse = reverse
        self.fill_remaining = fill_remaining

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        records = [record for dataset in data for record in dataset]
        universe = self._coverage_universe(records)
        ranked_records = self._rank_records(records)

        covered: set[str] = set()
        selected: list[Record] = []
        deferred: list[Record] = []

        for record in ranked_records:
            if self.target_size is not None and len(selected) >= self.target_size:
                break

            items = self._set_items(record)
            if items is None:
                if self.keep_missing:
                    deferred.append(record)
                continue

            new_items = items - covered
            if len(new_items) >= self.min_new_items:
                selected.append(record)
                covered |= items
            elif self.fill_remaining:
                deferred.append(record)

            if self.max_coverage_ratio is not None and universe:
                if len(covered) / len(universe) >= self.max_coverage_ratio:
                    break

        if self.target_size is not None and self.fill_remaining and len(selected) < self.target_size:
            for record in deferred:
                if len(selected) >= self.target_size:
                    break
                if record not in selected:
                    selected.append(record)
        elif self.target_size is None and self.fill_remaining:
            for record in deferred:
                if record not in selected:
                    selected.append(record)

        return selected

    def _coverage_universe(self, records: Sequence[Record]) -> set[str]:
        universe: set[str] = set()
        for record in records:
            items = self._set_items(record)
            if items is not None:
                universe |= items
        return universe

    def _rank_records(self, records: Sequence[Record]) -> list[Record]:
        if self.priority_key is None:
            return list(records)
        return sorted(
            records,
            key=lambda record: _coerce_numeric(self.get_field(record, self.priority_key, 0.0), self.priority_key),
            reverse=self.reverse,
        )

    def _set_items(self, record: Record) -> set[str] | None:
        raw_value = self.get_field(record, self.set_key, None)
        if raw_value is None:
            return None
        return _coerce_item_set(raw_value, self.set_key)


def _coerce_item_set(value: Any, set_key: str) -> set[str]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"Set field {set_key!r} must be a sequence of strings")
    items: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        items.add(text)
    if not items:
        raise ValueError(f"Set field {set_key!r} must not be empty")
    return items


def _coerce_numeric(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Priority field {key!r} must be numeric, got boolean")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Priority field {key!r} is not numeric: {value!r}") from exc
    raise ValueError(f"Priority field {key!r} is not numeric: {value!r}")
