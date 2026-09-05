"""Selection with atomic-group or per-group-cap semantics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

from dataos_operator.core import Dataset, SelectorOperator
from dataos_operator.core.contracts import get_field
from dataos_operator.core.selection import numeric, stable_fraction
from dataos_operator.view import DatasetView

GroupMode = Literal["atomic", "cap"]


class GroupAwareSelector(SelectorOperator):
    """Keep groups atomically or cap rows per group before an optional exact-size selection.

    Implementation: group by a stable field such as normalized-question hash or image hash;
    ``atomic`` preserves complete MCQ rotation families, while ``cap`` keeps at most the best N
    conversations per image. Ranking is score-first when supplied and seeded otherwise. Supports
    index-preserving ``DatasetView`` input; image packing/deduplication is never implicit.
    Dependency: pure-rule operator, no external dependencies.
    """

    dataset_backends = ("record", "indexed")

    def __init__(
        self,
        *,
        group_key: str,
        mode: GroupMode,
        target_size: int | None = None,
        max_per_group: int = 1,
        score_key: str | None = None,
        seed: int = 42,
        keep_missing: bool = True,
        require_exact: bool = True,
        batch_size: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            group_key=group_key,
            mode=mode,
            target_size=target_size,
            max_per_group=max_per_group,
            score_key=score_key,
            seed=seed,
            keep_missing=keep_missing,
            require_exact=require_exact,
            batch_size=batch_size,
            **kwargs,
        )
        if mode not in {"atomic", "cap"}:
            raise ValueError("mode must be atomic or cap")
        if target_size is not None and target_size <= 0:
            raise ValueError("target_size must be greater than zero")
        if max_per_group <= 0:
            raise ValueError("max_per_group must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.group_key = group_key
        self.mode = mode
        self.target_size = target_size
        self.max_per_group = max_per_group
        self.score_key = score_key
        self.seed = seed
        self.keep_missing = keep_missing
        self.require_exact = require_exact
        self.batch_size = batch_size

    def select(self, data: Sequence[Dataset]) -> Dataset:
        if len(data) == 1 and isinstance(data[0], DatasetView):
            return self._select_view(data[0])
        records = [record for dataset in data for record in dataset]
        entries = [
            (
                position,
                get_field(record, self.group_key, None),
                get_field(record, self.score_key, None) if self.score_key else None,
            )
            for position, record in enumerate(records)
        ]
        return [records[position] for position in self._select_entries(entries)]

    def _select_view(self, view: DatasetView) -> DatasetView:
        columns = [self.group_key, *([self.score_key] if self.score_key else [])]
        entries = []
        for batch in view.iter_batches(columns, batch_size=self.batch_size):
            for offset, position in enumerate(batch.positions):
                entries.append(
                    (
                        position,
                        batch[self.group_key][offset],
                        batch[self.score_key][offset] if self.score_key else None,
                    )
                )
        return view.select_positions(self._select_entries(entries))

    def _select_entries(self, entries: Sequence[tuple[int, Any, Any]]) -> list[int]:
        groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for position, raw_group, raw_score in entries:
            if raw_group is None or raw_group == "":
                if not self.keep_missing:
                    continue
                group = f"__missing__:{position}"
            else:
                group = str(raw_group)
            score = (
                numeric(raw_score, self.score_key)
                if self.score_key
                else stable_fraction(self.seed, position)
            )
            groups[group].append((score, position))

        if self.mode == "cap":
            candidates = [
                item
                for rows in groups.values()
                for item in sorted(rows, reverse=True)[: self.max_per_group]
            ]
            candidates.sort(reverse=True)
            target = self.target_size or len(candidates)
            selected = [position for _, position in candidates[:target]]
        else:
            ranked_groups = sorted(
                groups.items(),
                key=lambda item: (
                    max(score for score, _ in item[1]),
                    stable_fraction(self.seed, item[0]),
                ),
                reverse=True,
            )
            target = self.target_size or sum(len(rows) for rows in groups.values())
            selected = []
            for _, rows in ranked_groups:
                if len(selected) + len(rows) <= target:
                    selected.extend(position for _, position in rows)
                if len(selected) == target:
                    break

        if self.require_exact and len(selected) != target:
            raise ValueError(f"group constraints supplied {len(selected)} of {target} rows")
        return sorted(selected)
