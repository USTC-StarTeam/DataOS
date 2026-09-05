"""Exact deduplication over a composite record identity."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from dataos_operator.core import Dataset, Record, SelectorOperator
from dataos_operator.core.contracts import get_field, set_field
from dataos_operator.core.selection import numeric
from dataos_operator.view import DatasetView

KeepMode = Literal["first", "last", "top_score"]


class CompositeDeduplicateSelector(SelectorOperator):
    """Deduplicate by multiple fields while allowing shared images with distinct conversations.

    Implementation: canonicalize each configured field as deterministic JSON and keep the first,
    last, or highest-scored record for the resulting tuple. Use an image-content hash together
    with canonical conversation text for Curation-Bench; image-only grouping remains an explicit
    choice rather than a hidden default. Supports index-preserving ``DatasetView`` input.
    Dependency: pure-rule operator, no external dependencies.
    """

    dataset_backends = ("record", "indexed")

    def __init__(
        self,
        *,
        keys: Sequence[str],
        keep: KeepMode = "first",
        score_key: str | None = None,
        keep_missing: bool = True,
        batch_size: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            keys=keys,
            keep=keep,
            score_key=score_key,
            keep_missing=keep_missing,
            batch_size=batch_size,
            **kwargs,
        )
        if not keys:
            raise ValueError("keys must not be empty")
        if keep not in {"first", "last", "top_score"}:
            raise ValueError("keep must be first, last, or top_score")
        if keep == "top_score" and not score_key:
            raise ValueError("top_score requires score_key")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.keys = tuple(keys)
        self.keep = keep
        self.score_key = score_key
        self.keep_missing = keep_missing
        self.batch_size = batch_size

    def select(self, data: Sequence[Dataset]) -> Dataset:
        if len(data) == 1 and isinstance(data[0], DatasetView):
            return self._select_view(data[0])
        records = [record for dataset in data for record in dataset]
        positions = self._selected_positions(
            [(record, position) for position, record in enumerate(records)]
        )
        return [records[position] for position in positions]

    def _select_view(self, view: DatasetView) -> DatasetView:
        columns = [*self.keys, *([self.score_key] if self.score_key else [])]
        rows: list[tuple[Record, int]] = []
        for batch in view.iter_batches(columns, batch_size=self.batch_size):
            for offset, position in enumerate(batch.positions):
                record: Record = {}
                for key in columns:
                    record = set_field(record, key, batch[key][offset])
                rows.append((record, position))
        return view.select_positions(self._selected_positions(rows))

    def _selected_positions(self, rows: Sequence[tuple[Record, int]]) -> list[int]:
        chosen: dict[tuple[str, ...], tuple[int, float]] = {}
        missing: list[int] = []
        for record, position in rows:
            values = [get_field(record, key, None) for key in self.keys]
            if any(value is None for value in values):
                if self.keep_missing:
                    missing.append(position)
                continue
            identity = tuple(
                _canonical(value, key)
                for key, value in zip(self.keys, values, strict=True)
            )
            score = (
                numeric(get_field(record, self.score_key, None), self.score_key)
                if self.score_key
                else 0.0
            )
            previous = chosen.get(identity)
            if (
                previous is None
                or self.keep == "last"
                or (self.keep == "top_score" and score > previous[1])
            ):
                chosen[identity] = (position, score)
        return sorted([position for position, _ in chosen.values()] + missing)


def _canonical(value: Any, key: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise ValueError(
            f"Composite key {key!r} must be JSON-compatible; derive a stable hash first"
        ) from exc
