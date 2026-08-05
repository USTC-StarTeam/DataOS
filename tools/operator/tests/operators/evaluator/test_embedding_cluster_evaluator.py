from __future__ import annotations

from dataos_operator import Dataset
from dataos_operator.operators.evaluator.embedding_cluster_evaluator import EmbeddingClusterEvaluator


class StubEmbeddingClusterEvaluator(EmbeddingClusterEvaluator):
    def __init__(self, vectors: dict[str, list[float]], **kwargs):
        super().__init__(model="stub", pca_dim=None, normalize_embeddings=False, **kwargs)
        self._vectors = vectors

    def _compute_embedding(self, text: str) -> list[float]:
        return list(self._vectors[text])


def test_embedding_cluster_evaluator_writes_lightweight_cluster_metadata() -> None:
    dataset = Dataset.from_records(
        [
            {"query": "a", "answer": "1"},
            {"query": "b", "answer": "1"},
            {"query": "c", "answer": "1"},
            {"query": "d", "answer": "1"},
        ]
    )
    vectors = {
        "User: a\n\nAssistant: 1": [0.0, 0.0],
        "User: b\n\nAssistant: 1": [0.1, 0.0],
        "User: c\n\nAssistant: 1": [10.0, 10.0],
        "User: d\n\nAssistant: 1": [10.1, 10.0],
    }
    evaluator = StubEmbeddingClusterEvaluator(
        vectors=vectors,
        num_clusters=2,
        cluster_method="kmeans",
        random_state=0,
        max_iters=20,
    )

    clustered = evaluator.run(dataset)

    assert len(clustered) == 4
    cluster_ids = [record["metadata"]["cluster"]["id"] for record in clustered]
    assert set(cluster_ids) == {0, 1}
    for record in clustered:
        cluster_meta = record["metadata"]["cluster"]
        assert set(cluster_meta) == {"id", "size", "distance", "rank"}
        assert cluster_meta["size"] == 2
        assert cluster_meta["distance"] >= 0.0
        assert cluster_meta["rank"] in {1, 2}


def test_embedding_cluster_evaluator_resolves_cluster_count_from_target_cluster_size() -> None:
    dataset = Dataset.from_records(
        [
            {"text": "t0"},
            {"text": "t1"},
            {"text": "t2"},
            {"text": "t3"},
            {"text": "t4"},
        ]
    )
    vectors = {
        "t0": [0.0, 0.0],
        "t1": [0.1, 0.0],
        "t2": [5.0, 5.0],
        "t3": [5.1, 5.0],
        "t4": [9.0, 9.0],
    }
    evaluator = StubEmbeddingClusterEvaluator(
        vectors=vectors,
        text_key="text",
        target_cluster_size=2,
        cluster_method="kmeans",
        random_state=0,
        max_iters=20,
    )

    clustered = evaluator.run(dataset)

    cluster_ids = {record["metadata"]["cluster"]["id"] for record in clustered}
    assert len(cluster_ids) == 3
