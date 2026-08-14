"""Generate lightweight cluster metadata from text embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dataos_operator.core import EvaluatorOperator, Record
from dataos_operator.core.chat_formats import coerce_text, normalize_chat_messages


class EmbeddingClusterEvaluator(EvaluatorOperator):
    """Cluster samples by text and write lightweight cluster metadata (id/size/distance/rank).

    Implementation: a local HF encoder produces embeddings (optional L2 normalization and PCA
    reduction), then a self-implemented KMeans / MiniBatch-KMeans (pure numpy) clusters at the
    dataset level; only cluster metadata is written, not full vectors; reused when
    ``overwrite=False``.
    Dependency: local-model operator; loads a HF encoder on demand (transformers/torch) + numpy.
    """

    def __init__(
        self,
        backend: str = "local",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_params: dict[str, Any] | None = None,
        cluster_key: str = "metadata.cluster",
        text_template: str | None = None,
        messages_key: str = "messages",
        query_key: str = "query",
        answer_key: str = "answer",
        text_key: str = "text",
        normalize_embeddings: bool = True,
        pca_dim: int | None = 256,
        cluster_method: str = "kmeans",
        num_clusters: int | None = None,
        target_cluster_size: int = 1000,
        max_iters: int = 50,
        random_state: int = 42,
        batch_size: int = 1024,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backend=backend,
            model=model,
            model_params=model_params,
            cluster_key=cluster_key,
            text_template=text_template,
            messages_key=messages_key,
            query_key=query_key,
            answer_key=answer_key,
            text_key=text_key,
            normalize_embeddings=normalize_embeddings,
            pca_dim=pca_dim,
            cluster_method=cluster_method,
            num_clusters=num_clusters,
            target_cluster_size=target_cluster_size,
            max_iters=max_iters,
            random_state=random_state,
            batch_size=batch_size,
            overwrite=overwrite,
            **kwargs,
        )
        if backend != "local":
            raise ValueError("backend must be local in the first implementation")
        if cluster_method not in {"kmeans", "minibatch_kmeans"}:
            raise ValueError("cluster_method must be one of: kmeans, minibatch_kmeans")
        if pca_dim is not None and pca_dim <= 0:
            raise ValueError("pca_dim must be greater than zero")
        if num_clusters is not None and num_clusters <= 0:
            raise ValueError("num_clusters must be greater than zero")
        if target_cluster_size <= 0:
            raise ValueError("target_cluster_size must be greater than zero")
        if max_iters <= 0:
            raise ValueError("max_iters must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.backend = backend
        self.model = model
        self.model_params = dict(model_params or {})
        self.cluster_key = cluster_key
        self.text_template = text_template
        self.messages_key = messages_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.text_key = text_key
        self.normalize_embeddings = normalize_embeddings
        self.pca_dim = pca_dim
        self.cluster_method = cluster_method
        self.num_clusters = num_clusters
        self.target_cluster_size = target_cluster_size
        self.max_iters = max_iters
        self.random_state = random_state
        self.batch_size = batch_size
        self.overwrite = overwrite
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._max_length = int(self.model_params.pop("max_length", 512))

    def evaluate_record(self, record: Record) -> Record:
        return self.evaluate_dataset([record])[0]

    def evaluate_dataset(self, data: list[Record]) -> list[Record]:
        if not data:
            return []
        cluster_id_key = f"{self.cluster_key}.id"
        if not self.overwrite and all(self.get_field(record, cluster_id_key, None) is not None for record in data):
            return [dict(record) for record in data]

        texts = [self._cluster_text(record) for record in data]
        embeddings = np.asarray([self._compute_embedding(text) for text in texts], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(data):
            raise ValueError("embedding matrix shape is invalid")
        if self.normalize_embeddings:
            embeddings = _l2_normalize_rows(embeddings)

        reduced = _pca_reduce(embeddings, self.pca_dim)
        num_clusters = _resolve_num_clusters(
            num_records=len(data),
            requested_num_clusters=self.num_clusters,
            target_cluster_size=self.target_cluster_size,
        )
        num_clusters = min(num_clusters, len(data))

        if num_clusters == 1:
            assignments = np.zeros(len(data), dtype=np.int32)
            centroid = reduced.mean(axis=0, keepdims=True)
            distances = _euclidean_distance_rows(reduced, centroid[0])
        elif self.cluster_method == "minibatch_kmeans":
            assignments, distances = _fit_minibatch_kmeans(
                reduced,
                num_clusters=num_clusters,
                random_state=self.random_state,
                batch_size=self.batch_size,
                max_iters=self.max_iters,
            )
        else:
            assignments, distances = _fit_kmeans(
                reduced,
                num_clusters=num_clusters,
                random_state=self.random_state,
                max_iters=self.max_iters,
            )

        cluster_sizes = np.bincount(assignments, minlength=num_clusters)
        ranks = np.zeros(len(data), dtype=np.int32)
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(assignments == cluster_id)[0]
            if cluster_indices.size == 0:
                continue
            order = cluster_indices[np.argsort(distances[cluster_indices], kind="stable")]
            for rank, record_index in enumerate(order, start=1):
                ranks[record_index] = rank

        updated_records: list[Record] = []
        for index, record in enumerate(data):
            updated = self.set_field(record, f"{self.cluster_key}.id", int(assignments[index]))
            updated = self.set_field(updated, f"{self.cluster_key}.size", int(cluster_sizes[assignments[index]]))
            updated = self.set_field(updated, f"{self.cluster_key}.distance", float(distances[index]))
            updated = self.set_field(updated, f"{self.cluster_key}.rank", int(ranks[index]))
            updated_records.append(updated)
        return updated_records

    def _cluster_text(self, record: Record) -> str:
        messages_value = self.get_field(record, self.messages_key, None)
        if messages_value is not None:
            system_message, normalized_messages = normalize_chat_messages(messages_value, strict=True)
            parts: list[str] = []
            if system_message is not None:
                parts.append(f"System: {system_message}")
            for message in normalized_messages:
                role = "User" if message["role"] == "user" else "Assistant"
                parts.append(f"{role}: {message['content']}")
            text = "\n\n".join(parts).strip()
            if text:
                return text
            raise ValueError("messages must contain at least one usable message")

        if self.text_template is not None:
            text = self._format_template(self.text_template, record).strip()
            if text:
                return text
            raise ValueError("text_template produced empty text")

        query_value = self.get_field(record, self.query_key, None)
        answer_value = self.get_field(record, self.answer_key, None)
        query_text = coerce_text(query_value).strip() if query_value is not None else ""
        answer_text = coerce_text(answer_value).strip() if answer_value is not None else ""
        if query_text or answer_text:
            if query_text and answer_text:
                return f"User: {query_text}\n\nAssistant: {answer_text}"
            return query_text or answer_text

        text_value = self.get_field(record, self.text_key, None)
        if text_value is not None:
            text = coerce_text(text_value).strip()
            if text:
                return text
        raise KeyError(
            f"Missing clustering content: expected {self.messages_key!r}, text_template, "
            f"{self.query_key!r}/{self.answer_key!r}, or {self.text_key!r}"
        )

    def _format_template(self, template: str, record: Record) -> str:
        try:
            return template.format_map(_FlatRecordMapping(record))
        except KeyError as exc:
            raise KeyError(f"Template field is missing: {exc.args[0]}") from exc

    def _compute_embedding(self, text: str) -> list[float]:
        if not text:
            raise ValueError("clustering text must not be empty")
        self._ensure_model_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        device = next(self._model.parameters()).device
        with self._torch.no_grad():
            outputs = self._model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )
        hidden_state = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].to(device).unsqueeze(-1)
        masked_hidden = hidden_state * attention_mask
        token_count = attention_mask.sum(dim=1).clamp(min=1)
        pooled = masked_hidden.sum(dim=1) / token_count
        return [float(value) for value in pooled[0].detach().cpu().tolist()]

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        tokenizer_kwargs = {
            key: value
            for key, value in self.model_params.items()
            if key in {"trust_remote_code", "revision", "use_fast", "token", "local_files_only"}
        }
        model_kwargs = {
            key: value
            for key, value in self.model_params.items()
            if key
            in {
                "device_map",
                "torch_dtype",
                "trust_remote_code",
                "revision",
                "token",
                "local_files_only",
                "attn_implementation",
            }
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, **tokenizer_kwargs)
        self._model = AutoModel.from_pretrained(self.model, **model_kwargs)
        if "device_map" not in model_kwargs:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
        self._model.eval()


class _FlatRecordMapping(dict[str, Any]):
    def __init__(self, record: Mapping[str, Any]) -> None:
        super().__init__()
        self._record = record

    def __missing__(self, key: str) -> Any:
        if key in self._record:
            return self._record[key]
        raise KeyError(key)


def _resolve_num_clusters(
    *,
    num_records: int,
    requested_num_clusters: int | None,
    target_cluster_size: int,
) -> int:
    if requested_num_clusters is not None:
        return max(1, requested_num_clusters)
    return max(1, (num_records + target_cluster_size - 1) // target_cluster_size)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embedding vector norm must not be zero")
    return matrix / norms


def _pca_reduce(matrix: np.ndarray, pca_dim: int | None) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("embedding matrix must be 2D")
    if matrix.shape[0] <= 1:
        return matrix.astype(np.float32, copy=False)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if pca_dim is None:
        return centered.astype(np.float32, copy=False)
    target_dim = min(int(pca_dim), centered.shape[0], centered.shape[1])
    if target_dim <= 0:
        raise ValueError("pca_dim must be greater than zero after shape adjustment")
    if target_dim >= centered.shape[1]:
        return centered.astype(np.float32, copy=False)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:target_dim].T
    return (centered @ components).astype(np.float32, copy=False)


def _fit_kmeans(
    matrix: np.ndarray,
    *,
    num_clusters: int,
    random_state: int,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    centroids = _initialize_centroids(matrix, num_clusters, rng)
    assignments = np.full(matrix.shape[0], -1, dtype=np.int32)

    for _ in range(max_iters):
        distances = _pairwise_squared_distances(matrix, centroids)
        new_assignments = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(assignments, new_assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        for cluster_id in range(num_clusters):
            cluster_points = matrix[assignments == cluster_id]
            if cluster_points.size == 0:
                centroids[cluster_id] = matrix[rng.integers(0, matrix.shape[0])]
            else:
                centroids[cluster_id] = cluster_points.mean(axis=0)

    final_distances = _pairwise_squared_distances(matrix, centroids)
    assignments = np.argmin(final_distances, axis=1).astype(np.int32)
    distances = np.sqrt(final_distances[np.arange(matrix.shape[0]), assignments])
    return assignments, distances.astype(np.float32)


def _fit_minibatch_kmeans(
    matrix: np.ndarray,
    *,
    num_clusters: int,
    random_state: int,
    batch_size: int,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    centroids = _initialize_centroids(matrix, num_clusters, rng)
    counts = np.zeros(num_clusters, dtype=np.int64)

    for _ in range(max_iters):
        permutation = rng.permutation(matrix.shape[0])
        for start in range(0, matrix.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch = matrix[batch_indices]
            distances = _pairwise_squared_distances(batch, centroids)
            labels = np.argmin(distances, axis=1).astype(np.int32)
            for sample, label in zip(batch, labels, strict=True):
                counts[label] += 1
                rate = 1.0 / counts[label]
                centroids[label] = (1.0 - rate) * centroids[label] + rate * sample

    final_distances = _pairwise_squared_distances(matrix, centroids)
    assignments = np.argmin(final_distances, axis=1).astype(np.int32)
    distances = np.sqrt(final_distances[np.arange(matrix.shape[0]), assignments])
    return assignments, distances.astype(np.float32)


def _initialize_centroids(matrix: np.ndarray, num_clusters: int, rng: np.random.Generator) -> np.ndarray:
    if num_clusters > matrix.shape[0]:
        raise ValueError("num_clusters must not exceed the number of records")
    indices = rng.choice(matrix.shape[0], size=num_clusters, replace=False)
    return matrix[indices].copy()


def _pairwise_squared_distances(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - centroids[None, :, :]
    return np.sum(diff * diff, axis=2)


def _euclidean_distance_rows(matrix: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    diff = matrix - centroid
    return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)
