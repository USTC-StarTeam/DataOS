"""Select a balanced subset from cluster assignments."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import random

from dataos_operator.core import Dataset, Record, SelectorOperator


class ClusterBalanceSelector(SelectorOperator):
    """Assign per-cluster quotas from existing cluster labels for balanced sampling.

    Implementation: group by the ``cluster_key`` id, allocate per-cluster quotas via
    ``quota_strategy`` (uniform/sqrt/proportional, with remainder top-up), and pick within each
    cluster by ``intra_cluster_strategy`` (top_score/centroid_first/random); missing cluster
    ids are handled per ``keep_missing``.
    Dependency: pure-rule operator, no external dependencies; requires an upstream clustering
    operator (e.g. embedding_cluster_evaluator) to produce cluster ids.
    """

    def __init__(
        self,
        target_size: int,
        cluster_key: str = "metadata.cluster.id",
        score_key: str | None = None,
        distance_key: str = "metadata.cluster.distance",
        quota_strategy: str = "sqrt",
        intra_cluster_strategy: str = "top_score",
        keep_missing: bool = False,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_size=target_size,
            cluster_key=cluster_key,
            score_key=score_key,
            distance_key=distance_key,
            quota_strategy=quota_strategy,
            intra_cluster_strategy=intra_cluster_strategy,
            keep_missing=keep_missing,
            random_state=random_state,
            **kwargs,
        )
        if target_size <= 0:
            raise ValueError("target_size must be greater than zero")
        if quota_strategy not in {"uniform", "sqrt", "proportional"}:
            raise ValueError("quota_strategy must be one of: uniform, sqrt, proportional")
        if intra_cluster_strategy not in {"top_score", "random", "centroid_first"}:
            raise ValueError("intra_cluster_strategy must be one of: top_score, random, centroid_first")

        self.target_size = target_size
        self.cluster_key = cluster_key
        self.score_key = score_key
        self.distance_key = distance_key
        self.quota_strategy = quota_strategy
        self.intra_cluster_strategy = intra_cluster_strategy
        self.keep_missing = keep_missing
        self.random_state = random_state

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        cluster_groups: dict[int, list[Record]] = defaultdict(list)
        missing_records: list[Record] = []

        for dataset in data:
            for record in dataset:
                raw_cluster = self.get_field(record, self.cluster_key, None)
                if raw_cluster is None:
                    if self.keep_missing:
                        missing_records.append(record)
                    continue
                cluster_groups[_coerce_cluster_id(raw_cluster, self.cluster_key)].append(record)

        if not cluster_groups:
            return list(missing_records[: self.target_size])

        ordered_groups = {
            cluster_id: self._order_cluster_records(cluster_groups[cluster_id])
            for cluster_id in sorted(cluster_groups)
        }
        total_available = sum(len(records) for records in ordered_groups.values())
        if total_available <= self.target_size:
            selected_all: list[Record] = []
            for cluster_id in sorted(ordered_groups):
                selected_all.extend(ordered_groups[cluster_id])
            if self.keep_missing and len(selected_all) < self.target_size:
                selected_all.extend(missing_records[: self.target_size - len(selected_all)])
            return selected_all

        quotas = _compute_cluster_quotas(
            cluster_sizes={cluster_id: len(records) for cluster_id, records in ordered_groups.items()},
            target_size=self.target_size,
            quota_strategy=self.quota_strategy,
        )

        selected: list[Record] = []
        remaining: list[Record] = []
        for cluster_id in sorted(ordered_groups):
            records = ordered_groups[cluster_id]
            quota = quotas.get(cluster_id, 0)
            selected.extend(records[:quota])
            remaining.extend(records[quota:])

        if len(selected) < self.target_size:
            remaining = self._order_remaining_records(remaining)
            selected.extend(remaining[: self.target_size - len(selected)])

        if self.keep_missing and len(selected) < self.target_size:
            selected.extend(missing_records[: self.target_size - len(selected)])
        return selected[: self.target_size]

    def _order_cluster_records(self, records: list[Record]) -> list[Record]:
        ordered = list(records)
        strategy = self.intra_cluster_strategy
        if strategy == "top_score" and self.score_key is not None:
            ordered.sort(
                key=lambda record: _numeric_or_default(
                    self.get_field(record, self.score_key, None),
                    self.score_key,
                    default=float("-inf"),
                ),
                reverse=True,
            )
            return ordered
        if strategy == "centroid_first":
            ordered.sort(
                key=lambda record: _numeric_or_default(
                    self.get_field(record, self.distance_key, None),
                    self.distance_key,
                    default=float("inf"),
                )
            )
            return ordered
        rng = random.Random(self.random_state)
        rng.shuffle(ordered)
        return ordered

    def _order_remaining_records(self, records: list[Record]) -> list[Record]:
        ordered = list(records)
        if self.score_key is not None:
            ordered.sort(
                key=lambda record: _numeric_or_default(
                    self.get_field(record, self.score_key, None),
                    self.score_key,
                    default=float("-inf"),
                ),
                reverse=True,
            )
            return ordered
        if self.intra_cluster_strategy == "centroid_first":
            ordered.sort(
                key=lambda record: _numeric_or_default(
                    self.get_field(record, self.distance_key, None),
                    self.distance_key,
                    default=float("inf"),
                )
            )
            return ordered
        rng = random.Random(self.random_state)
        rng.shuffle(ordered)
        return ordered


def _compute_cluster_quotas(
    *,
    cluster_sizes: dict[int, int],
    target_size: int,
    quota_strategy: str,
) -> dict[int, int]:
    weights = {
        cluster_id: _quota_weight(size, quota_strategy)
        for cluster_id, size in cluster_sizes.items()
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("cluster weights must sum to a positive value")

    raw_quotas = {
        cluster_id: target_size * (weight / total_weight)
        for cluster_id, weight in weights.items()
    }
    quotas = {
        cluster_id: min(cluster_sizes[cluster_id], int(raw_quota))
        for cluster_id, raw_quota in raw_quotas.items()
    }
    assigned = sum(quotas.values())
    if assigned >= target_size:
        return quotas

    remainders = sorted(
        (
            raw_quotas[cluster_id] - int(raw_quotas[cluster_id]),
            cluster_sizes[cluster_id],
            cluster_id,
        )
        for cluster_id in cluster_sizes
    )
    remainders.reverse()

    while assigned < target_size:
        progressed = False
        for _, _, cluster_id in remainders:
            if quotas[cluster_id] >= cluster_sizes[cluster_id]:
                continue
            quotas[cluster_id] += 1
            assigned += 1
            progressed = True
            if assigned >= target_size:
                break
        if not progressed:
            break
    return quotas


def _quota_weight(cluster_size: int, quota_strategy: str) -> float:
    if cluster_size <= 0:
        raise ValueError("cluster_size must be greater than zero")
    if quota_strategy == "uniform":
        return 1.0
    if quota_strategy == "sqrt":
        return math.sqrt(cluster_size)
    return float(cluster_size)


def _coerce_cluster_id(value: Any, cluster_key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Cluster field {cluster_key!r} must not be boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
    raise ValueError(f"Cluster field {cluster_key!r} is not an integer id: {value!r}")


def _numeric_or_default(value: Any, field_key: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"Field {field_key!r} must be numeric, got boolean")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Field {field_key!r} is not numeric: {value!r}") from exc
    raise ValueError(f"Field {field_key!r} is not numeric: {value!r}")
