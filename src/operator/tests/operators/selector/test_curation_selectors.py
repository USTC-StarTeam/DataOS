from __future__ import annotations

from collections import Counter

from datasets import Dataset as HFDataset

from dataos_operator import Dataset, DatasetView
from dataos_operator.operators.selector import (
    AnchorRefreshSelector,
    CompositeDeduplicateSelector,
    GroupAwareSelector,
    StratifiedQuotaSelector,
    WeightedReservoirSelector,
)


def _records() -> list[dict]:
    return [
        {
            "id": index,
            "subset": "a" if index < 6 else "b",
            "score": float(index),
            "weight": float(index + 1),
        }
        for index in range(12)
    ]


def test_weighted_reservoir_is_deterministic_and_honors_strata_quotas() -> None:
    operator = WeightedReservoirSelector(
        sample_size=5,
        weight_key="weight",
        strata_keys=["subset"],
        quotas={"a": 2, "b": 3},
        seed=7,
    )

    first = operator.run(Dataset.from_records(_records())).to_records()
    second = operator.run(Dataset.from_records(_records())).to_records()

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert Counter(row["subset"] for row in first) == {"a": 2, "b": 3}


def test_stratified_quota_selects_top_scores_on_dataset_view() -> None:
    source = HFDataset.from_list(_records())
    result = StratifiedQuotaSelector(
        strata_keys=["subset"],
        quotas={"a": 2, "b": 2},
        score_key="score",
    ).run(DatasetView(source))

    assert isinstance(result, DatasetView)
    assert result.source["id"] == list(range(12))
    assert [result[position]["id"] for position in range(len(result))] == [4, 5, 10, 11]


def test_stratified_quota_allocates_capacity_weighted_target() -> None:
    records = _records()
    result = StratifiedQuotaSelector(
        strata_keys=["subset"],
        target_size=6,
        strata_weights={"a": 1.0, "b": 2.0},
        seed=3,
    ).run(Dataset.from_records(records))

    assert Counter(row["subset"] for row in result) == {"a": 2, "b": 4}


def test_composite_dedup_allows_same_image_with_distinct_text() -> None:
    rows = [
        {"id": 0, "image_hash": "x", "text_hash": "a", "score": 1},
        {"id": 1, "image_hash": "x", "text_hash": "b", "score": 1},
        {"id": 2, "image_hash": "x", "text_hash": "a", "score": 2},
    ]
    result = CompositeDeduplicateSelector(
        keys=["image_hash", "text_hash"],
        keep="top_score",
        score_key="score",
    ).run(Dataset.from_records(rows))

    assert [row["id"] for row in result] == [1, 2]


def test_group_cap_and_atomic_modes() -> None:
    rows = [
        {"id": 0, "group": "q1", "score": 1},
        {"id": 1, "group": "q1", "score": 3},
        {"id": 2, "group": "q2", "score": 2},
        {"id": 3, "group": "q2", "score": 4},
    ]
    capped = GroupAwareSelector(
        group_key="group", mode="cap", max_per_group=1, score_key="score"
    ).run(Dataset.from_records(rows))
    atomic = GroupAwareSelector(
        group_key="group", mode="atomic", target_size=2, score_key="score"
    ).run(Dataset.from_records(rows))

    assert [row["id"] for row in capped] == [1, 3]
    assert [row["id"] for row in atomic] == [2, 3]


def test_anchor_refresh_preserves_reference_strata_and_exact_anchor_dose() -> None:
    rows = []
    for index in range(16):
        rows.append(
            {
                "id": index,
                "subset": "a" if index < 8 else "b",
                "metadata": {"reference": index in {0, 1, 2, 3, 8, 9, 10, 11}},
                "score": float(index),
            }
        )
    result = AnchorRefreshSelector(
        strata_keys=["subset"],
        retain_fraction=0.5,
        score_key="score",
        seed=9,
        exclude_reference_replacements=True,
    ).run(Dataset.from_records(rows))

    assert len(result) == 8
    assert Counter(row["subset"] for row in result) == {"a": 4, "b": 4}
    assert sum(bool(row["metadata"]["reference"]) for row in result) == 4
