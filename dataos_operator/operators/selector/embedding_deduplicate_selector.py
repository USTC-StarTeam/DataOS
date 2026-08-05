"""按 embedding 相似度去重的 selector。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Dataset, Record, SelectorOperator


class EmbeddingDeduplicateSelector(SelectorOperator):
    """Filter semantically duplicate samples by embedding cosine similarity.

    Implementation: read the ``embedding_key`` vector and scan greedily in order—drop a record
    as duplicate when its cosine similarity to any kept sample is ``similarity_threshold`` or
    higher; optional L2 normalization and a ``max_samples`` cap.
    Dependency: pure-rule operator (pure Python, no external dependencies); requires an
    upstream embedding operator to produce vectors.
    """

    def __init__(
        self,
        embedding_key: str = "metadata.embedding.vector",
        similarity_metric: str = "cosine",
        similarity_threshold: float = 0.95,
        keep_missing: bool = False,
        normalize_embeddings: bool = True,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embedding_key=embedding_key,
            similarity_metric=similarity_metric,
            similarity_threshold=similarity_threshold,
            keep_missing=keep_missing,
            normalize_embeddings=normalize_embeddings,
            max_samples=max_samples,
            **kwargs,
        )
        if similarity_metric != "cosine":
            raise ValueError("similarity_metric must be cosine in the first implementation")
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be greater than zero")

        self.embedding_key = embedding_key
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.keep_missing = keep_missing
        self.normalize_embeddings = normalize_embeddings
        self.max_samples = max_samples

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        selected: list[Record] = []
        selected_vectors: list[list[float]] = []

        for dataset in data:
            for record in dataset:
                if self.max_samples is not None and len(selected) >= self.max_samples:
                    return selected

                raw_vector = self.get_field(record, self.embedding_key, None)
                if raw_vector is None:
                    if self.keep_missing:
                        selected.append(record)
                    continue

                vector = _coerce_vector(raw_vector, self.embedding_key)
                if self.normalize_embeddings:
                    vector = _l2_normalize(vector)

                is_duplicate = any(
                    _cosine_similarity(vector, kept_vector) >= self.similarity_threshold
                    for kept_vector in selected_vectors
                )
                if is_duplicate:
                    continue

                selected.append(record)
                selected_vectors.append(vector)
        return selected


def _coerce_vector(value: Any, embedding_key: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"Embedding field {embedding_key!r} must be a numeric sequence")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"Embedding field {embedding_key!r} must not contain booleans")
        if not isinstance(item, int | float):
            raise ValueError(f"Embedding field {embedding_key!r} must contain numeric values")
        vector.append(float(item))
    if not vector:
        raise ValueError(f"Embedding field {embedding_key!r} must not be empty")
    return vector


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("embedding vector norm must not be zero")
    return [value / norm for value in vector]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimensionality")
    return sum(l * r for l, r in zip(left, right, strict=True))
