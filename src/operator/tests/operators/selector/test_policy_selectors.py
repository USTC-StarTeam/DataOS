from __future__ import annotations

from dataos_operator import Dataset
from dataos_operator.runtime import ForwardSignal, PolicyRuntime, RuntimeContext
from dataos_operator.runtime.types import BatchView
from dataos_operator.operators.selector import (
    DeltaLossSelector,
    LessSelector,
    LossSelector,
    NearSelector,
    NiceSelector,
    TsdsSelector,
)


class FakeBackend:
    def mean_gradient(self, records, *, projection_dim=None, projection_seed=42):  # noqa: ANN001, ARG002
        vectors = [self.gradient_vector(record, projection_dim=projection_dim, projection_seed=projection_seed) for record in records]
        width = len(vectors[0])
        return [sum(vector[column] for vector in vectors) / len(vectors) for column in range(width)]

    def gradient_vector(self, record, *, projection_dim=None, projection_seed=42):  # noqa: ANN001, ARG002
        identifier = record.get("id")
        answer = record.get("answer") or record.get("output")
        if identifier in {"eval-a", "train-a"}:
            return [1.0, 0.0]
        if identifier == "train-b":
            return [0.8, 0.2]
        if identifier == "train-c":
            return [0.0, 1.0]
        if answer == "correct":
            return [1.0, 0.0]
        return [0.0, 1.0]

    def generate_candidates(self, record, *, num_samples=4, temperature=0.7):  # noqa: ANN001, ARG002
        del num_samples, temperature
        return ["correct", "incorrect"]

    def reward_score(self, record, candidate):  # noqa: ANN001
        expected = record.get("answer") or record.get("output")
        return 1.0 if candidate == expected else 0.0


def test_loss_selector_prefers_high_loss_records() -> None:
    operator = LossSelector(sample_size=2, focus="high", focus_weight=20.0, random_state=7)

    result = operator.run(
        [
            {"id": "a", "metadata": {"runtime": {"loss": 0.1}}},
            {"id": "b", "metadata": {"runtime": {"loss": 0.2}}},
            {"id": "c", "metadata": {"runtime": {"loss": 0.9}}},
            {"id": "d", "metadata": {"runtime": {"loss": 1.1}}},
        ]
    )

    assert {record["id"] for record in result} == {"c", "d"}


def test_delta_loss_selector_uses_runtime_losses() -> None:
    dataset = Dataset.from_records(
        [
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ]
    )
    batch = BatchView(records=dataset.to_records(), indices=[0, 1, 2, 3], domain_ids=["x", "x", "y", "y"])
    runtime = PolicyRuntime(
        [DeltaLossSelector(sample_size=2, window_fraction=0.5, random_state=11)],
        context=RuntimeContext(mode="online", step=2, metadata={"total_updates": 4}),
    )
    runtime.after_forward(batch, ForwardSignal(per_sample_losses=[1.0, 0.9, 0.3, 0.2]))
    runtime.after_forward(batch, ForwardSignal(per_sample_losses=[0.4, 0.8, 0.25, 0.2]))
    action = runtime.on_update_boundary(candidates=dataset.to_records(), total_updates=4)[-1]

    assert action.selected_indices is not None
    assert len(action.selected_indices) == 2
    assert set(action.selected_indices).issubset({0, 1, 2, 3})
    assert action.metadata["progress"] == 0.5


def test_near_selector_round_robins_query_neighbors() -> None:
    queries = Dataset.from_records(
        [
            {"id": "q1", "metadata": {"embedding": {"vector": [1.0, 0.0]}}},
            {"id": "q2", "metadata": {"embedding": {"vector": [0.0, 1.0]}}},
        ]
    )
    candidates = Dataset.from_records(
        [
            {"id": "a", "metadata": {"embedding": {"vector": [1.0, 0.0]}}},
            {"id": "b", "metadata": {"embedding": {"vector": [0.9, 0.1]}}},
            {"id": "c", "metadata": {"embedding": {"vector": [0.0, 1.0]}}},
            {"id": "d", "metadata": {"embedding": {"vector": [0.1, 0.9]}}},
        ]
    )

    result = NearSelector(queries, sample_size=3, max_neighbors=3).run(candidates)

    assert [record["id"] for record in result] == ["a", "c", "b"]


def test_tsds_selector_returns_requested_count() -> None:
    queries = Dataset.from_records(
        [
            {"id": "q1", "metadata": {"embedding": {"vector": [1.0, 0.0]}}},
            {"id": "q2", "metadata": {"embedding": {"vector": [0.0, 1.0]}}},
        ]
    )
    candidates = Dataset.from_records(
        [
            {"id": "a", "metadata": {"embedding": {"vector": [1.0, 0.0]}}},
            {"id": "b", "metadata": {"embedding": {"vector": [0.9, 0.1]}}},
            {"id": "c", "metadata": {"embedding": {"vector": [0.0, 1.0]}}},
            {"id": "d", "metadata": {"embedding": {"vector": [0.1, 0.9]}}},
        ]
    )

    result = TsdsSelector(
        queries,
        sample_size=3,
        max_neighbors=3,
        kde_neighbors=2,
        sigma=0.8,
        alpha=0.5,
        budget=3.0,
        random_state=13,
    ).run(candidates)

    assert len(result) == 3
    assert len({record["id"] for record in result}) == 3


def test_less_selector_prepare_caches_eval_gradient() -> None:
    backend = FakeBackend()
    eval_dataset = Dataset.from_records([{"id": "eval-a", "answer": "correct"}])
    candidates = Dataset.from_records(
        [
            {"id": "train-a", "answer": "correct"},
            {"id": "train-b", "answer": "correct"},
            {"id": "train-c", "answer": "incorrect"},
        ]
    )
    runtime = PolicyRuntime(
        [LessSelector(eval_dataset, sample_size=2, projection_dim=None)],
        context=RuntimeContext(mode="hybrid", metadata={"backend": backend}),
    )

    runtime.prepare()
    action = runtime.on_update_boundary(candidates=candidates.to_records(), backend=backend)[-1]

    assert runtime.states["less_selector"].data["eval_gradient"] == [1.0, 0.0]
    assert action.selected_indices == [0, 1]


def test_nice_selector_uses_reward_weighted_eval_gradient() -> None:
    backend = FakeBackend()
    eval_dataset = Dataset.from_records([{"id": "eval-a", "answer": "correct"}])
    candidates = Dataset.from_records(
        [
            {"id": "train-a", "answer": "correct"},
            {"id": "train-b", "answer": "correct"},
            {"id": "train-c", "answer": "incorrect"},
        ]
    )
    runtime = PolicyRuntime(
        [NiceSelector(eval_dataset, sample_size=2, projection_dim=None, reward_samples=2)],
        context=RuntimeContext(mode="hybrid", metadata={"backend": backend}),
    )

    runtime.prepare()
    action = runtime.on_update_boundary(candidates=candidates.to_records(), backend=backend)[-1]

    assert action.selected_indices == [0, 1]
