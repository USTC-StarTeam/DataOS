from __future__ import annotations

from dataos_operator import Dataset
from dataos_operator.operators.selector.cluster_balance_selector import ClusterBalanceSelector


def test_cluster_balance_selector_balances_uniformly_across_clusters() -> None:
    dataset = Dataset.from_records(
        [
            {"id": "a1", "metadata": {"cluster": {"id": 0, "distance": 0.3}}},
            {"id": "a2", "metadata": {"cluster": {"id": 0, "distance": 0.1}}},
            {"id": "a3", "metadata": {"cluster": {"id": 0, "distance": 0.2}}},
            {"id": "b1", "metadata": {"cluster": {"id": 1, "distance": 0.4}}},
            {"id": "b2", "metadata": {"cluster": {"id": 1, "distance": 0.2}}},
            {"id": "b3", "metadata": {"cluster": {"id": 1, "distance": 0.1}}},
        ]
    )
    selector = ClusterBalanceSelector(
        target_size=4,
        quota_strategy="uniform",
        intra_cluster_strategy="centroid_first",
    )

    selected = selector.run(dataset)

    assert len(selected) == 4
    counts = {}
    for record in selected:
        cluster_id = record["metadata"]["cluster"]["id"]
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    assert counts == {0: 2, 1: 2}
    assert [record["id"] for record in selected if record["metadata"]["cluster"]["id"] == 0] == ["a2", "a3"]
    assert [record["id"] for record in selected if record["metadata"]["cluster"]["id"] == 1] == ["b3", "b2"]


def test_cluster_balance_selector_prefers_high_scores_with_fill() -> None:
    dataset = Dataset.from_records(
        [
            {"id": "a1", "metadata": {"cluster": {"id": 0, "distance": 0.3}, "score": 0.3}},
            {"id": "a2", "metadata": {"cluster": {"id": 0, "distance": 0.1}, "score": 0.9}},
            {"id": "b1", "metadata": {"cluster": {"id": 1, "distance": 0.4}, "score": 0.4}},
            {"id": "b2", "metadata": {"cluster": {"id": 1, "distance": 0.2}, "score": 0.8}},
            {"id": "c1", "metadata": {"cluster": {"id": 2, "distance": 0.3}, "score": 0.7}},
        ]
    )
    selector = ClusterBalanceSelector(
        target_size=4,
        quota_strategy="sqrt",
        score_key="metadata.score",
        intra_cluster_strategy="top_score",
    )

    selected = selector.run(dataset)

    assert len(selected) == 4
    selected_ids = [record["id"] for record in selected]
    assert {"a2", "b2", "c1"}.issubset(selected_ids)
    assert "a1" not in selected_ids
