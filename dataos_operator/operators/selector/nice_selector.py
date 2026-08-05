"""NICE selector with reward-aware evaluation trajectory gradients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dataos_operator import Dataset
from dataos_operator.core import Record, SelectorOperator
from dataos_operator.runtime import OperatorState, PolicyAction, RuntimeContext


class NiceSelector(SelectorOperator):
    """Approximate NICE with reward-weighted policy gradients on evaluation samples.

    Implementation: like LESS, but the evaluation gradient is reward-weighted—for each eval sample
    the backend generates ``reward_samples`` candidates at ``temperature``, scores them with
    ``reward_score``, and forms a reward-normalized weighted sum of their random-projected
    gradients; candidates are then scored by cosine similarity to this eval gradient and the top
    ``sample_size`` selected. The offline ``.run()`` path ranks records by a precomputed
    ``score_key``.
    Dependency: model-in-the-loop operator; the online path requires a backend with
    ``generate_candidates``/``reward_score``/``gradient_vector`` in ``context.metadata["backend"]``
    plus numpy; the offline ``.run()`` path needs precomputed scores at ``score_key``.
    """

    def __init__(
        self,
        eval_datasets: Dataset | Sequence[Dataset],
        *,
        sample_size: int,
        score_key: str = "metadata.nice.score",
        projection_dim: int | None = 256,
        projection_seed: int = 42,
        reward_samples: int = 4,
        temperature: float = 0.7,
        eval_sample_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            eval_datasets=eval_datasets,
            sample_size=sample_size,
            score_key=score_key,
            projection_dim=projection_dim,
            projection_seed=projection_seed,
            reward_samples=reward_samples,
            temperature=temperature,
            eval_sample_size=eval_sample_size,
            **kwargs,
        )
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")
        if reward_samples <= 0:
            raise ValueError("reward_samples must be greater than zero")

        self.eval_datasets = _normalize_datasets(eval_datasets)
        self.sample_size = sample_size
        self.score_key = score_key
        self.projection_dim = projection_dim
        self.projection_seed = projection_seed
        self.reward_samples = reward_samples
        self.temperature = temperature
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
            raise ValueError("NICE evaluation gradient norm must be positive")
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
        reward_gradients = [
            self._reward_weighted_gradient(backend, record, seed_offset=index)
            for index, record in enumerate(eval_records)
        ]
        return np.mean(np.stack(reward_gradients, axis=0), axis=0).astype(np.float32).tolist()

    def _reward_weighted_gradient(self, backend: Any, record: Record, *, seed_offset: int) -> np.ndarray:
        raw_candidates = backend.generate_candidates(
            record,
            num_samples=self.reward_samples,
            temperature=self.temperature,
        )
        candidates = [candidate.strip() for candidate in raw_candidates if candidate.strip()]
        if not candidates:
            prompt_target = backend.prompt_target(record)
            if prompt_target.target.strip():
                candidates = [prompt_target.target.strip()]
            else:
                raise ValueError("NICE requires at least one non-empty candidate or target")
        rewards = np.asarray([backend.reward_score(record, candidate) for candidate in candidates], dtype=np.float32)
        if np.allclose(rewards.sum(), 0.0):
            rewards = np.ones_like(rewards) / len(rewards)
        else:
            rewards = rewards / rewards.sum()

        gradients: list[np.ndarray] = []
        for candidate_index, candidate in enumerate(candidates):
            pseudo_record = _replace_target(record, candidate)
            gradient = np.asarray(
                backend.gradient_vector(
                    pseudo_record,
                    projection_dim=self.projection_dim,
                    projection_seed=self.projection_seed + seed_offset * 97 + candidate_index,
                ),
                dtype=np.float32,
            )
            gradients.append(gradient)
        stacked = np.stack(gradients, axis=0)
        weighted = np.sum(stacked * rewards[:, None], axis=0)
        norm = np.linalg.norm(weighted)
        if norm == 0.0:
            return weighted
        return weighted / norm


def _normalize_datasets(value: Dataset | Sequence[Dataset]) -> list[Dataset]:
    if isinstance(value, Dataset):
        return [value]
    return list(value)


def _replace_target(record: Record, candidate: str) -> Record:
    updated = dict(record)
    if "output" in updated:
        updated["output"] = candidate
    elif "answer" in updated:
        updated["answer"] = candidate
    elif "messages" in updated:
        messages = list(updated["messages"])
        if not messages:
            raise ValueError("messages must not be empty")
        last_message = dict(messages[-1])
        last_message["content"] = candidate
        messages[-1] = last_message
        updated["messages"] = messages
    elif "text" in updated:
        updated["text"] = candidate
    else:
        raise KeyError("Cannot replace target in record without output/answer/messages/text")
    return updated
