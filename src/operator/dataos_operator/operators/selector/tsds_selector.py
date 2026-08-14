"""TSDS selector with query-aware probability estimation from embedding neighborhoods."""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from dataos_operator import Dataset
from dataos_operator.core import Record, SelectorOperator


class TsdsSelector(SelectorOperator):
    """Sample from TSDS probabilities computed over query-candidate neighborhoods.

    Implementation: from candidate/query embeddings (``embedding_key``) build each query's
    nearest-neighbor list (at most ``max_neighbors``), estimate a local KDE density per candidate
    (``kde_neighbors`` neighbors, bandwidth ``sigma``), then run the TSDS cost-bounded heap
    allocation (relevance/diversity trade-off ``alpha``, cost ``budget``) to derive a sampling
    distribution and draw ``sample_size`` records without replacement (seeded by ``random_state``).
    Dependency: pure-rule operator (pure Python + numpy, no model); requires an upstream embedding
    operator to produce vectors. Offline only (no runtime hooks).
    """

    def __init__(
        self,
        query_datasets: Dataset | Sequence[Dataset],
        *,
        sample_size: int,
        embedding_key: str = "metadata.embedding.vector",
        max_neighbors: int = 500,
        kde_neighbors: int = 128,
        sigma: float = 0.75,
        alpha: float = 0.6,
        budget: float = 5.0,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            query_datasets=query_datasets,
            sample_size=sample_size,
            embedding_key=embedding_key,
            max_neighbors=max_neighbors,
            kde_neighbors=kde_neighbors,
            sigma=sigma,
            alpha=alpha,
            budget=budget,
            random_state=random_state,
            **kwargs,
        )
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")
        if max_neighbors <= 1:
            raise ValueError("max_neighbors must be greater than one")
        if kde_neighbors <= 0:
            raise ValueError("kde_neighbors must be greater than zero")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")

        self.query_datasets = _normalize_query_datasets(query_datasets)
        self.sample_size = sample_size
        self.embedding_key = embedding_key
        self.max_neighbors = max_neighbors
        self.kde_neighbors = kde_neighbors
        self.sigma = sigma
        self.alpha = alpha
        self.budget = budget
        self.random_state = random_state

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
        probabilities = self._compute_probabilities(candidate_vectors, query_vectors)
        rng = random.Random(self.random_state)
        available = list(range(len(candidates)))
        weights = probabilities.tolist()
        selected_indices: list[int] = []
        while available and len(selected_indices) < self.sample_size:
            total = sum(weights)
            threshold = rng.random() * total if total > 0 else 0.0
            running = 0.0
            choice = len(available) - 1
            for position, weight in enumerate(weights):
                running += weight
                if running >= threshold:
                    choice = position
                    break
            selected_indices.append(available.pop(choice))
            weights.pop(choice)
        return [candidates[index] for index in selected_indices]

    def _compute_probabilities(self, candidates: np.ndarray, queries: np.ndarray) -> np.ndarray:
        similarity = queries @ candidates.T
        neighbor_order = np.argsort(-similarity, axis=1)
        max_neighbors = min(self.max_neighbors, neighbor_order.shape[1])
        top_indices = neighbor_order[:, :max_neighbors]
        top_distances = 1.0 - similarity[np.arange(queries.shape[0])[:, None], top_indices]

        if self.sigma == 0:
            top_kdes = np.ones_like(top_indices, dtype=np.float32)
        else:
            unique_ids = np.unique(top_indices.reshape(-1))
            sub_candidates = candidates[unique_ids]
            sub_similarity = sub_candidates @ sub_candidates.T
            sub_distance = 1.0 - sub_similarity
            neighbor_count = min(self.kde_neighbors, sub_distance.shape[1])
            local_order = np.argsort(sub_distance, axis=1)[:, :neighbor_count]
            kernel_sum = np.zeros(sub_candidates.shape[0], dtype=np.float32)
            for row_index in range(sub_candidates.shape[0]):
                row_neighbors = local_order[row_index]
                row_distances = sub_distance[row_index, row_neighbors]
                kernel = np.maximum(0.0, 1.0 - row_distances / (self.sigma ** 2))
                kernel_sum[row_index] = float(kernel.sum())
            kde_map = {int(unique_ids[row]): kernel_sum[row] for row in range(len(unique_ids))}
            top_kdes = np.vectorize(lambda item: kde_map[int(item)])(top_indices)

        last_neighbor = [0 for _ in range(queries.shape[0])]
        heap: list[tuple[float, int, int]] = [
            (1.0 / max(top_kdes[row, 0], 1e-12), 0, row)
            for row in range(queries.shape[0])
        ]
        heapq.heapify(heap)
        running_cost = np.zeros(queries.shape[0], dtype=np.float32)
        distance_weighted_sum = np.asarray(
            [top_distances[row, 0] / max(top_kdes[row, 0], 1e-12) for row in range(queries.shape[0])],
            dtype=np.float32,
        )
        total_cost = 0.0
        scale = 0.0

        while heap:
            count, neighbor_index, query_index = heapq.heappop(heap)
            scale = count
            total_cost -= float(running_cost[query_index])
            running_cost[query_index] = float(
                top_distances[query_index, neighbor_index + 1] * count - distance_weighted_sum[query_index]
            )
            total_cost += float(running_cost[query_index])
            if self.alpha / self.budget * total_cost >= (1 - self.alpha) * queries.shape[0]:
                break
            last_neighbor[query_index] = neighbor_index
            if neighbor_index < max_neighbors - 2:
                count += 1.0 / max(top_kdes[query_index, neighbor_index + 1], 1e-12)
                heapq.heappush(heap, (count, neighbor_index + 1, query_index))
                distance_weighted_sum[query_index] += (
                    top_distances[query_index, neighbor_index + 1]
                    / max(top_kdes[query_index, neighbor_index + 1], 1e-12)
                )

        probabilities = np.zeros(candidates.shape[0], dtype=np.float64)
        inv_query_count = 1.0 / queries.shape[0]
        for query_index in range(queries.shape[0]):
            local_sum = 0.0
            for neighbor_index in range(last_neighbor[query_index] + 1):
                weight = inv_query_count / max(scale, 1e-12) / max(top_kdes[query_index, neighbor_index], 1e-12)
                probabilities[top_indices[query_index, neighbor_index]] += weight
                local_sum += weight
            next_neighbor = min(last_neighbor[query_index] + 1, max_neighbors - 1)
            probabilities[top_indices[query_index, next_neighbor]] += max(inv_query_count - local_sum, 0.0)

        probabilities = np.maximum(probabilities, 0.0)
        if probabilities.sum() == 0.0:
            probabilities = np.repeat(1.0 / len(probabilities), len(probabilities))
        else:
            probabilities /= probabilities.sum()
        return probabilities


def _normalize_query_datasets(query_datasets: Dataset | Sequence[Dataset]) -> list[Dataset]:
    if isinstance(query_datasets, Dataset):
        return [query_datasets]
    return list(query_datasets)


def _coerce_vector(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("Embedding must be a numeric sequence")
    vector = [float(item) for item in value]
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0.0:
        raise ValueError("Embedding vector norm must be positive")
    return [item / norm for item in vector]
