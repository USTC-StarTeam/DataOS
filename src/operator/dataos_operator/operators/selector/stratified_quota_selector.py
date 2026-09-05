"""Capacity-aware exact quota selection over arbitrary strata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core import Dataset, SelectorOperator
from dataos_operator.core.contracts import get_field
from dataos_operator.core.selection import (
    allocate_quotas,
    numeric,
    stable_fraction,
    stratum_label,
)
from dataos_operator.view import DatasetView


class StratifiedQuotaSelector(SelectorOperator):
    """Select exact explicit or capacity-weighted quotas across arbitrary record strata.

    Implementation: group by ``strata_keys`` (labels are ``|``-joined values), allocate an exact
    largest-remainder-style target under real capacities, then choose seeded-random or top-scored
    rows inside each stratum. Explicit quotas reproduce source/format cells; ``strata_weights``
    supports task-level allocation such as VisionFlan's answer-length-weighted task mix. Supports
    index-preserving ``DatasetView`` input and deterministic global backfill.
    Dependency: pure-rule operator, no external dependencies.
    """

    dataset_backends = ("record", "indexed")

    def __init__(
        self,
        *,
        strata_keys: Sequence[str],
        target_size: int | None = None,
        quotas: Mapping[str, int] | None = None,
        strata_weights: Mapping[str, float] | None = None,
        score_key: str | None = None,
        descending: bool = True,
        seed: int = 42,
        minimum_per_stratum: int = 0,
        backfill: bool = True,
        batch_size: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            strata_keys=strata_keys,
            target_size=target_size,
            quotas=quotas,
            strata_weights=strata_weights,
            score_key=score_key,
            descending=descending,
            seed=seed,
            minimum_per_stratum=minimum_per_stratum,
            backfill=backfill,
            batch_size=batch_size,
            **kwargs,
        )
        explicit = dict(quotas or {})
        if not strata_keys:
            raise ValueError("strata_keys must not be empty")
        if target_size is None and not explicit:
            raise ValueError("target_size or quotas must be provided")
        if target_size is not None and target_size <= 0:
            raise ValueError("target_size must be greater than zero")
        if any(value < 0 for value in explicit.values()):
            raise ValueError("quotas must be non-negative")
        if explicit and target_size is not None and sum(explicit.values()) != target_size:
            raise ValueError("target_size must equal the sum of explicit quotas")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.strata_keys = tuple(strata_keys)
        self.target_size = target_size or sum(explicit.values())
        self.quotas = explicit
        self.strata_weights = dict(strata_weights or {})
        self.score_key = score_key
        self.descending = descending
        self.seed = seed
        self.minimum_per_stratum = minimum_per_stratum
        self.backfill = backfill
        self.batch_size = batch_size

    def select(self, data: Sequence[Dataset]) -> Dataset:
        if len(data) == 1 and isinstance(data[0], DatasetView):
            return self._select_view(data[0])
        records = [record for dataset in data for record in dataset]
        entries = [
            self._entry(
                position,
                [get_field(record, key, None) for key in self.strata_keys],
                get_field(record, self.score_key, None) if self.score_key else None,
            )
            for position, record in enumerate(records)
        ]
        return [records[position] for position in self._select_entries(entries)]

    def _select_view(self, view: DatasetView) -> DatasetView:
        columns = [*self.strata_keys, *([self.score_key] if self.score_key else [])]
        entries = []
        for batch in view.iter_batches(columns, batch_size=self.batch_size):
            for offset, position in enumerate(batch.positions):
                entries.append(
                    self._entry(
                        position,
                        [batch[key][offset] for key in self.strata_keys],
                        batch[self.score_key][offset] if self.score_key else None,
                    )
                )
        return view.select_positions(self._select_entries(entries))

    def _entry(
        self,
        position: int,
        strata: Sequence[Any],
        raw_score: Any,
    ) -> tuple[int, str, float]:
        score = numeric(raw_score, self.score_key) if self.score_key else 0.0
        rank_score = score if self.descending else -score
        tie = stable_fraction(self.seed, position)
        return position, stratum_label(strata), rank_score + tie * 1e-12

    def _select_entries(self, entries: Sequence[tuple[int, str, float]]) -> list[int]:
        groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for position, label, rank in entries:
            groups[label].append((rank, position))
        capacities = {label: len(rows) for label, rows in groups.items()}
        quotas = self.quotas or allocate_quotas(
            capacities,
            self.target_size,
            weights=self.strata_weights,
            minimum_per_stratum=self.minimum_per_stratum,
        )

        selected: set[int] = set()
        for label, quota in quotas.items():
            rows = sorted(groups.get(label, []), reverse=True)
            selected.update(position for _, position in rows[:quota])

        if len(selected) < self.target_size and self.backfill:
            remaining = sorted(
                (
                    (rank, position)
                    for rows in groups.values()
                    for rank, position in rows
                    if position not in selected
                ),
                reverse=True,
            )
            for _, position in remaining:
                selected.add(position)
                if len(selected) == self.target_size:
                    break
        if len(selected) != self.target_size:
            raise ValueError(f"strata supplied {len(selected)} of {self.target_size} required rows")
        return sorted(selected)
