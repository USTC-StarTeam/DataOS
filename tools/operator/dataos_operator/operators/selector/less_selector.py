"""LESS selector based on gradient similarity to evaluation data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dataos_operator import Dataset
from dataos_operator.core import Record, SelectorOperator
from dataos_operator.runtime import OperatorState, PolicyAction, RuntimeContext


class LessSelector(SelectorOperator):
    """Approximate LESS by matching train-sample gradients to evaluation gradients.

    Implementation: ``prepare`` computes a mean evaluation gradient over ``eval_datasets`` via the
    model backend; at each update boundary score every candidate by the cosine similarity between
    its random-projected (dim ``projection_dim``, seed ``projection_seed``) gradient and the
    normalized eval gradient, then pick the top ``sample_size``. The offline ``.run()`` path
    instead ranks records by a precomputed ``score_key``.
    Dependency: model-in-the-loop operator; the online path requires a gradient-capable backend in
    ``context.metadata["backend"]`` (``gradient_vector``/``mean_gradient``) plus numpy; the offline
    ``.run()`` path needs precomputed scores at ``score_key``.
    """

    def __init__(
        self,
        eval_datasets: Dataset | Sequence[Dataset],
        *,
        sample_size: int,
        score_key: str = "metadata.less.score",
        projection_dim: int | None = 256,
        projection_seed: int = 42,
        eval_sample_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            eval_datasets=eval_datasets,
            sample_size=sample_size,
            score_key=score_key,
            projection_dim=projection_dim,
            projection_seed=projection_seed,
            eval_sample_size=eval_sample_size,
            **kwargs,
        )
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")

        self.eval_datasets = _normalize_datasets(eval_datasets)
        self.sample_size = sample_size
        self.score_key = score_key
        self.projection_dim = projection_dim
        self.projection_seed = projection_seed
        self.eval_sample_size = eval_sample_size

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        records = [record for dataset in data for record in dataset]
        ranked = sorted(
            (
                (float(self.require_field(record, self.score_key)), record)
                for record in records
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [record for _, record in ranked[: self.sample_size]]

    def prepare(
        self,
        *,
        datasets: Mapping[str, Dataset] | None,
        state: OperatorState,
        context: RuntimeContext,
    ) -> tuple[OperatorState, PolicyAction | None]:
        del datasets
        backend = context.metadata.get("backend")
        if backend is None:
            return state, None
        state.data["eval_gradient"] = self._evaluation_gradient(backend)
        return state, None

    def on_update_boundary(
        self,
        *,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        del context
        candidates = kwargs.get("candidates")
        backend = kwargs.get("backend")
        if candidates is None or backend is None:
            return None
        eval_gradient = state.data.get("eval_gradient")
        if eval_gradient is None:
            eval_gradient = self._evaluation_gradient(backend)
            state.data["eval_gradient"] = eval_gradient
        scores = self.score_records(candidates, backend, eval_gradient=eval_gradient)
        state.data["scores"] = scores
        selected = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[: self.sample_size]
        return PolicyAction(
            selected_indices=selected,
            metadata={"selector": self.name, "scores": scores},
        )

    def score_records(
        self,
        records: Sequence[Record],
        backend: Any,
        *,
        eval_gradient: Sequence[float] | None = None,
    ) -> list[float]:
        eval_gradient_array = np.asarray(
            eval_gradient if eval_gradient is not None else self._evaluation_gradient(backend),
            dtype=np.float32,
        )
        eval_norm = np.linalg.norm(eval_gradient_array)
        if eval_norm == 0.0:
            raise ValueError("Evaluation gradient norm must be positive")
        eval_gradient_array = eval_gradient_array / eval_norm

        scores: list[float] = []
        for index, record in enumerate(records):
            gradient = np.asarray(
                backend.gradient_vector(
                    record,
                    projection_dim=self.projection_dim,
                    projection_seed=self.projection_seed + index + 1,
                ),
                dtype=np.float32,
            )
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm == 0.0:
                scores.append(0.0)
                continue
            scores.append(float(np.dot(gradient / gradient_norm, eval_gradient_array)))
        return scores

    def _evaluation_gradient(self, backend: Any) -> list[float]:
        eval_records = [record for dataset in self.eval_datasets for record in dataset]
        if self.eval_sample_size is not None:
            eval_records = eval_records[: self.eval_sample_size]
        return list(
            backend.mean_gradient(
                eval_records,
                projection_dim=self.projection_dim,
                projection_seed=self.projection_seed,
            )
        )


def _normalize_datasets(value: Dataset | Sequence[Dataset]) -> list[Dataset]:
    if isinstance(value, Dataset):
        return [value]
    return list(value)
