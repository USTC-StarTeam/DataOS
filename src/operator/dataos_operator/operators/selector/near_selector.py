"""Nearest-example round-robin selector for distribution-aware data retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from dataos_operator import Dataset
from dataos_operator.core import Record, SelectorOperator


class NearSelector(SelectorOperator):
    """Select samples by interleaving nearest neighbors for each query example.

    Implementation: read normalized embedding vectors (``embedding_key``) for the candidates and
    the ``query_datasets``, compute query-candidate cosine similarity, then round-robin across
    queries—each round picks each query's next nearest still-unused candidate—until ``sample_size``
    unique records are gathered (scanning at most ``max_neighbors`` ranks per query); raises if too
    few unique neighbors are reachable.
    Dependency: pure-rule operator (pure Python + numpy, no model); requires an upstream embedding
    operator to produce vectors. Offline only (no runtime hooks).
    """

    def __init__(
        self,
        query_datasets: Dataset | Sequence[Dataset],
        *,
        sample_size: int,
        embedding_key: str = "metadata.embedding.vector",
        max_neighbors: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            query_datasets=query_datasets,
            sample_size=sample_size,
            embedding_key=embedding_key,
            max_neighbors=max_neighbors,
            **kwargs,
        )
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be greater than zero")

        self.query_datasets = _normalize_query_datasets(query_datasets)
        self.sample_size = sample_size
        self.embedding_key = embedding_key
        self.max_neighbors = max_neighbors

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        candidates = [record for dataset in data for record in dataset]
        candidate_vectors = np.asarray(
            [_coerce_vector(self.require_field(record, self.embedding_key)) for record in candidates],
            dtype=np.float32,
        )
        query_records = [record for dataset in self.query_datasets for record in dataset]
        query_vectors = np.asarray(
            [_coerce_vector(self.require_field(record, self.embedding_key)) for record in query_records],
            dtype=np.float32,
        )

        similarity = query_vectors @ candidate_vectors.T
        neighbor_order = np.argsort(-similarity, axis=1)
        max_neighbors = min(self.max_neighbors, neighbor_order.shape[1])

        selected_ids: list[int] = []
        selected_set: set[int] = set()
        for neighbor_rank in range(max_neighbors):
            for query_index in range(neighbor_order.shape[0]):
                candidate_index = int(neighbor_order[query_index, neighbor_rank])
                if candidate_index in selected_set:
                    continue
                selected_set.add(candidate_index)
                selected_ids.append(candidate_index)
                if len(selected_ids) >= self.sample_size:
                    return [candidates[index] for index in selected_ids]

        if len(selected_ids) < self.sample_size:
            raise ValueError(
                f"Unable to select {self.sample_size} unique samples from {len(selected_ids)} reachable neighbors"
            )
        return [candidates[index] for index in selected_ids]


def _normalize_query_datasets(query_datasets: Dataset | Sequence[Dataset]) -> list[Dataset]:
    if isinstance(query_datasets, Sequence) and query_datasets and isinstance(query_datasets[0], Dataset):
        return list(query_datasets)  # pragma: no cover - RuntimeDataset isinstance is awkward here
    if isinstance(query_datasets, Dataset):
        return [query_datasets]
    return list(query_datasets) if isinstance(query_datasets, Sequence) else [query_datasets]


def _coerce_vector(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("Embedding must be a numeric sequence")
    vector = [float(item) for item in value]
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0.0:
        raise ValueError("Embedding vector norm must be positive")
    return [item / norm for item in vector]
