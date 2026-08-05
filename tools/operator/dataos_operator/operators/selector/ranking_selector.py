"""按排序键选择样本的通用 selector。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Dataset, Record, SelectorOperator


class RankingSelector(SelectorOperator):
    """Rank by one or more numeric fields and select the top-k or bottom-k records.

    Implementation: read ``sort_keys`` into a sort tuple, order by ``reverse`` (asc/desc),
    then take top_k or bottom_k; records missing the sort field may backfill via
    ``keep_missing`` when the quota is not met.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        sort_keys: str | Sequence[str],
        *,
        top_k: int | None = None,
        bottom_k: int | None = None,
        reverse: bool = True,
        keep_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sort_keys=sort_keys,
            top_k=top_k,
            bottom_k=bottom_k,
            reverse=reverse,
            keep_missing=keep_missing,
            **kwargs,
        )
        normalized_sort_keys = (sort_keys,) if isinstance(sort_keys, str) else tuple(sort_keys)
        if not normalized_sort_keys:
            raise ValueError("sort_keys cannot be empty")
        if top_k is None and bottom_k is None:
            raise ValueError("top_k and bottom_k cannot both be None")
        if top_k is not None and bottom_k is not None:
            raise ValueError("top_k and bottom_k cannot both be set")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if bottom_k is not None and bottom_k <= 0:
            raise ValueError("bottom_k must be greater than zero")

        self.sort_keys = normalized_sort_keys
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.reverse = reverse
        self.keep_missing = keep_missing

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        ranked_records: list[tuple[tuple[float, ...], Record]] = []
        missing_records: list[Record] = []

        for dataset in data:
            for record in dataset:
                sort_values = self._sort_values(record)
                if sort_values is None:
                    if self.keep_missing:
                        missing_records.append(record)
                    continue
                ranked_records.append((sort_values, record))

        ranked_records.sort(key=lambda item: item[0], reverse=self.reverse)
        sorted_records = [record for _, record in ranked_records]

        if self.top_k is not None:
            selected = list(sorted_records[: self.top_k])
            if self.keep_missing and len(selected) < self.top_k:
                selected.extend(missing_records[: self.top_k - len(selected)])
            return selected

        assert self.bottom_k is not None
        selected = list(sorted_records[-self.bottom_k :])
        if not self.reverse:
            selected = list(sorted_records[: self.bottom_k])
        if self.keep_missing and len(selected) < self.bottom_k:
            selected.extend(missing_records[: self.bottom_k - len(selected)])
        return selected

    def _sort_values(self, record: Record) -> tuple[float, ...] | None:
        values: list[float] = []
        for key in self.sort_keys:
            raw_value = self.get_field(record, key, None)
            if raw_value is None:
                return None
            values.append(_coerce_numeric(raw_value, key))
        return tuple(values)


def _coerce_numeric(value: Any, sort_key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Sort field {sort_key!r} must be numeric, got boolean")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Sort field {sort_key!r} is not numeric: {value!r}") from exc
    raise ValueError(f"Sort field {sort_key!r} is not numeric: {value!r}")
