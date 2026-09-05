"""Controlled within-stratum refresh of a reference selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Dataset, SelectorOperator
from dataos_operator.core.contracts import get_field
from dataos_operator.core.selection import allocate_quotas, numeric, stable_fraction, stratum_label
from dataos_operator.view import DatasetView


class AnchorRefreshSelector(SelectorOperator):
    """Retain a fixed reference fraction and replace the rest inside identical strata.

    Implementation: derive exact per-stratum quotas from rows marked by ``reference_key``, allocate
    the requested global anchor count proportionally, retain deterministic reference anchors, and
    fill each stratum by score plus seeded tie-break. This captures the successful conservative
    Curation-Bench intervention while leaving source/format mixture unchanged. Supports
    index-preserving ``DatasetView`` input.
    Dependency: pure-rule operator, no external dependencies.
    """

    dataset_backends = ("record", "indexed")

    def __init__(
        self,
        *,
        reference_key: str = "metadata.reference",
        strata_keys: Sequence[str],
        retain_fraction: float,
        score_key: str | None = None,
        seed: int = 42,
        exclude_reference_replacements: bool = False,
        batch_size: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            reference_key=reference_key,
            strata_keys=strata_keys,
            retain_fraction=retain_fraction,
            score_key=score_key,
            seed=seed,
            exclude_reference_replacements=exclude_reference_replacements,
            batch_size=batch_size,
            **kwargs,
        )
        if not strata_keys:
            raise ValueError("strata_keys must not be empty")
        if not 0 <= retain_fraction <= 1:
            raise ValueError("retain_fraction must be between zero and one")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.reference_key = reference_key
        self.strata_keys = tuple(strata_keys)
        self.retain_fraction = retain_fraction
        self.score_key = score_key
        self.seed = seed
        self.exclude_reference_replacements = exclude_reference_replacements
        self.batch_size = batch_size

    def select(self, data: Sequence[Dataset]) -> Dataset:
        if len(data) != 1:
            raise ValueError("anchor refresh accepts one pool dataset")
        if isinstance(data[0], DatasetView):
            return self._select_view(data[0])
        records = list(data[0])
        entries = [
            (
                position,
                bool(get_field(record, self.reference_key, False)),
                stratum_label([get_field(record, key, None) for key in self.strata_keys]),
                get_field(record, self.score_key, None) if self.score_key else None,
            )
            for position, record in enumerate(records)
        ]
        return [records[position] for position in self._select_entries(entries)]

    def _select_view(self, view: DatasetView) -> DatasetView:
        columns = [
            self.reference_key,
            *self.strata_keys,
            *([self.score_key] if self.score_key else []),
        ]
        entries = []
        for batch in view.iter_batches(columns, batch_size=self.batch_size):
            for offset, position in enumerate(batch.positions):
                entries.append(
                    (
                        position,
                        bool(batch[self.reference_key][offset]),
                        stratum_label([batch[key][offset] for key in self.strata_keys]),
                        batch[self.score_key][offset] if self.score_key else None,
                    )
                )
        return view.select_positions(self._select_entries(entries))

    def _select_entries(
        self,
        entries: Sequence[tuple[int, bool, str, Any]],
    ) -> list[int]:
        reference: dict[str, list[int]] = defaultdict(list)
        by_stratum: dict[str, list[tuple[float, float, int, bool]]] = defaultdict(list)
        for position, is_reference, label, raw_score in entries:
            if is_reference:
                reference[label].append(position)
            score = numeric(raw_score, self.score_key) if self.score_key else 0.0
            tie = stable_fraction(self.seed, position)
            by_stratum[label].append((score, tie, position, is_reference))

        reference_counts = {label: len(rows) for label, rows in reference.items()}
        target_size = sum(reference_counts.values())
        retain_size = round(target_size * self.retain_fraction)
        retain_quotas = allocate_quotas(reference_counts, retain_size)

        selected: set[int] = set()
        for label, quota in reference_counts.items():
            anchors = sorted(
                reference[label], key=lambda position: stable_fraction(self.seed, position)
            )[: retain_quotas[label]]
            local_selected = set(anchors)
            replacements = [
                row
                for row in by_stratum.get(label, [])
                if row[2] not in local_selected
                and not (self.exclude_reference_replacements and row[3])
            ]
            replacements.sort(reverse=True)
            needed = quota - len(anchors)
            local_selected.update(position for _, _, position, _ in replacements[:needed])
            if len(local_selected) != quota:
                raise ValueError(f"stratum {label!r} cannot supply its reference quota {quota}")
            selected.update(local_selected)
        if len(selected) != target_size:
            raise ValueError(f"anchor refresh selected {len(selected)} of {target_size} rows")
        return sorted(selected)
