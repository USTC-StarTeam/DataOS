"""Streaming weighted reservoir sampling with optional exact strata quotas."""

from __future__ import annotations

import heapq
import random
from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core import Dataset, SelectorOperator
from dataos_operator.core.contracts import get_field
from dataos_operator.core.selection import numeric, stratum_label
from dataos_operator.view import DatasetView


class WeightedReservoirSelector(SelectorOperator):
    """Select a bounded weighted reservoir globally or independently within exact strata quotas.

    Implementation: use the Efraimidis-Spirakis key ``U ** (1 / weight)`` and bounded min-heaps;
    optional ``quotas`` are keyed by ``|``-joined stratum values. A global eligible reservoir can
    backfill under-capacity strata to an exact sample size. ``DatasetView`` scanning reads only
    weight/strata columns and returns original row indices, matching the dominant high-performing
    Curation-Bench baseline algorithm.
    Dependency: pure-rule operator, no external dependencies.
    """

    dataset_backends = ("record", "indexed")

    def __init__(
        self,
        *,
        sample_size: int | None = None,
        weight_key: str | None = None,
        strata_keys: Sequence[str] = (),
        quotas: Mapping[str, int] | None = None,
        seed: int = 42,
        backfill: bool = True,
        batch_size: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_size=sample_size,
            weight_key=weight_key,
            strata_keys=strata_keys,
            quotas=quotas,
            seed=seed,
            backfill=backfill,
            batch_size=batch_size,
            **kwargs,
        )
        normalized_quotas = dict(quotas or {})
        if sample_size is None and not normalized_quotas:
            raise ValueError("sample_size or quotas must be provided")
        if sample_size is not None and sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")
        if any(value < 0 for value in normalized_quotas.values()):
            raise ValueError("quotas must be non-negative")
        quota_total = sum(normalized_quotas.values())
        if sample_size is not None and normalized_quotas and sample_size != quota_total:
            raise ValueError("sample_size must equal the sum of quotas")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.sample_size = sample_size or quota_total
        self.weight_key = weight_key
        self.strata_keys = tuple(strata_keys)
        self.quotas = normalized_quotas
        self.seed = seed
        self.backfill = backfill
        self.batch_size = batch_size

    def select(self, data: Sequence[Dataset]) -> Dataset:
        if len(data) == 1 and isinstance(data[0], DatasetView):
            return self._select_view(data[0])
        records = [record for dataset in data for record in dataset]
        entries = (
            (
                position,
                get_field(record, self.weight_key, 1.0) if self.weight_key else 1.0,
                [get_field(record, key, None) for key in self.strata_keys],
            )
            for position, record in enumerate(records)
        )
        selected = self._sample(entries)
        return [records[position] for position in selected]

    def _select_view(self, view: DatasetView) -> DatasetView:
        columns = [*self.strata_keys, *([self.weight_key] if self.weight_key else [])]

        def entries():
            for batch in view.iter_batches(columns, batch_size=self.batch_size):
                for offset, position in enumerate(batch.positions):
                    weight = batch[self.weight_key][offset] if self.weight_key else 1.0
                    strata = [batch[key][offset] for key in self.strata_keys]
                    yield position, weight, strata

        return view.select_positions(self._sample(entries()))

    def _sample(self, entries) -> list[int]:
        rng = random.Random(self.seed)
        strata_heaps: dict[str, list[tuple[float, int]]] = {label: [] for label in self.quotas}
        global_heap: list[tuple[float, int]] = []
        eligible = 0
        for position, raw_weight, strata in entries:
            weight = numeric(raw_weight, self.weight_key or "weight")
            if weight <= 0:
                continue
            label = stratum_label(strata)
            if self.quotas and label not in self.quotas:
                continue
            eligible += 1
            key = rng.random() ** (1.0 / weight)
            _heap_push(global_heap, (key, position), self.sample_size)
            if self.quotas:
                _heap_push(strata_heaps[label], (key, position), self.quotas[label])

        if eligible < self.sample_size:
            raise ValueError(
                f"only {eligible} positive-weight rows are eligible for {self.sample_size} slots"
            )
        selected = (
            {position for heap in strata_heaps.values() for _, position in heap}
            if self.quotas
            else {position for _, position in global_heap}
        )
        if len(selected) < self.sample_size and self.backfill:
            for _, position in sorted(global_heap, reverse=True):
                selected.add(position)
                if len(selected) == self.sample_size:
                    break
        if len(selected) != self.sample_size:
            raise ValueError(f"strata supplied {len(selected)} of {self.sample_size} required rows")
        return sorted(selected)


def _heap_push(heap: list[tuple[float, int]], item: tuple[float, int], size: int) -> None:
    if size <= 0:
        return
    if len(heap) < size:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)
