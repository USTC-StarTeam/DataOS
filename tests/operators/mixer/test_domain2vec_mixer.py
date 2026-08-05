from __future__ import annotations

from dataos_operator import Dataset
from dataos_operator.operators.mixer.domain2vec_mixer import Domain2VecMixer


class StubDomain2VecMixer(Domain2VecMixer):
    def __init__(self, vectors: dict[str, list[float]], **kwargs):
        super().__init__(model="stub", **kwargs)
        self._vectors = vectors

    def _compute_embedding(self, text: str) -> list[float]:
        return list(self._vectors[text])


def test_domain2vec_mixer_prefers_target_like_domain() -> None:
    math_dataset = Dataset.from_records(
        [
            {"text": "math-1", "metadata": {"source": "math"}},
            {"text": "math-2", "metadata": {"source": "math"}},
            {"text": "math-3", "metadata": {"source": "math"}},
            {"text": "math-4", "metadata": {"source": "math"}},
        ]
    )
    chat_dataset = Dataset.from_records(
        [
            {"text": "chat-1", "metadata": {"source": "chat"}},
            {"text": "chat-2", "metadata": {"source": "chat"}},
            {"text": "chat-3", "metadata": {"source": "chat"}},
            {"text": "chat-4", "metadata": {"source": "chat"}},
        ]
    )
    target_dataset = Dataset.from_records(
        [
            {"text": "target-1"},
            {"text": "target-2"},
            {"text": "target-3"},
        ]
    )
    vectors = {
        "math-1": [1.0, 0.0],
        "math-2": [0.9, 0.1],
        "math-3": [1.0, 0.1],
        "math-4": [0.95, 0.05],
        "chat-1": [0.0, 1.0],
        "chat-2": [0.1, 0.9],
        "chat-3": [0.1, 1.0],
        "chat-4": [0.05, 0.95],
        "target-1": [1.0, 0.0],
        "target-2": [0.95, 0.05],
        "target-3": [0.9, 0.1],
    }
    mixer = StubDomain2VecMixer(
        vectors=vectors,
        target_datasets=target_dataset,
        domain_key="metadata.source",
        text_key="text",
        output_size=6,
        source_sample_size=3,
        target_sample_size=3,
        temperature=0.5,
        random_state=7,
    )

    mixed = mixer.run([math_dataset, chat_dataset])

    assert len(mixed) == 6
    sources = [record["metadata"]["source"] for record in mixed]
    assert sources.count("math") > sources.count("chat")
    metadata = mixed.metadata["domain2vec"]
    assert metadata["domain_key"] == "metadata.source"
    assert set(metadata["source_domains"]) == {"math", "chat"}
    assert metadata["mixture_weights"]["math"] > metadata["mixture_weights"]["chat"]
    assert metadata["target_dataset_count"] == 1
    assert metadata["source_sample_size"] == 3
    assert metadata["target_sample_size"] == 3


def test_domain2vec_mixer_accepts_cluster_id_as_domain_key() -> None:
    source_dataset = Dataset.from_records(
        [
            {"text": "cluster-a1", "metadata": {"cluster": {"id": 0}}},
            {"text": "cluster-a2", "metadata": {"cluster": {"id": 0}}},
            {"text": "cluster-b1", "metadata": {"cluster": {"id": 1}}},
            {"text": "cluster-b2", "metadata": {"cluster": {"id": 1}}},
        ]
    )
    target_dataset = Dataset.from_records(
        [
            {"text": "target-a"},
            {"text": "target-b"},
        ]
    )
    vectors = {
        "cluster-a1": [1.0, 0.0],
        "cluster-a2": [0.9, 0.0],
        "cluster-b1": [0.0, 1.0],
        "cluster-b2": [0.1, 1.0],
        "target-a": [1.0, 0.0],
        "target-b": [0.9, 0.1],
    }
    mixer = StubDomain2VecMixer(
        vectors=vectors,
        target_datasets=target_dataset,
        domain_key="metadata.cluster.id",
        text_key="text",
        output_size=4,
        source_sample_size=2,
        target_sample_size=2,
        random_state=11,
    )

    mixed = mixer.run([source_dataset])

    assert len(mixed) == 4
    domain_meta = mixed.metadata["domain2vec"]
    assert set(domain_meta["source_domains"]) == {"0", "1"}


def test_domain2vec_mixer_supports_multiple_target_datasets() -> None:
    source_a = Dataset.from_records([
        {"text": "a1", "metadata": {"source": "a"}},
        {"text": "a2", "metadata": {"source": "a"}},
    ])
    source_b = Dataset.from_records([
        {"text": "b1", "metadata": {"source": "b"}},
        {"text": "b2", "metadata": {"source": "b"}},
    ])
    target_1 = Dataset.from_records([{"text": "ta1"}])
    target_2 = Dataset.from_records([{"text": "tb1"}])
    vectors = {
        "a1": [1.0, 0.0],
        "a2": [1.0, 0.1],
        "b1": [0.0, 1.0],
        "b2": [0.1, 1.0],
        "ta1": [1.0, 0.0],
        "tb1": [0.0, 1.0],
    }
    mixer = StubDomain2VecMixer(
        vectors=vectors,
        target_datasets=[target_1, target_2],
        domain_key="metadata.source",
        text_key="text",
        output_size=4,
        source_sample_size=2,
        target_sample_size=2,
        random_state=3,
    )

    mixed = mixer.run([source_a, source_b])

    assert len(mixed) == 4
    metadata = mixed.metadata["domain2vec"]
    assert metadata["target_dataset_count"] == 2
    assert set(metadata["source_domains"]) == {"a", "b"}
