"""Loss-based selector inspired by hard-sample focused training curricula."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Dataset, Record, SelectorOperator
from dataos_operator.runtime import OperatorState, PolicyAction, RuntimeContext


class LossSelector(SelectorOperator):
    """Select records with a controllable bias toward low/medium/high loss samples.

    Implementation: split the losses into low/medium/high buckets by ``quantiles``, give every
    record base weight 1.0 and multiply the ``focus`` bucket by ``focus_weight``, raise the
    weights to the power ``1/temperature``, then weighted-sample ``sample_size`` records (with or
    without replacement per ``replacement``, seeded by ``random_state``); records missing a loss
    are dropped unless ``keep_missing``. Losses come from ``loss_key`` in the offline ``.run()``
    path, or are accumulated across steps via the runtime ``after_forward`` hook for online use.
    Dependency: pure-rule operator, no external dependencies; online use requires per-sample
    losses fed through the PolicyRuntime, offline ``.run()`` reads them from record metadata.
    """

    def __init__(
        self,
        sample_size: int,
        *,
        loss_key: str = "metadata.runtime.loss",
        focus: str = "high",
        focus_weight: float = 5.0,
        quantiles: tuple[float, float] = (0.33, 0.66),
        replacement: bool = False,
        temperature: float = 1.0,
        random_state: int = 42,
        keep_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_size=sample_size,
            loss_key=loss_key,
            focus=focus,
            focus_weight=focus_weight,
            quantiles=quantiles,
            replacement=replacement,
            temperature=temperature,
            random_state=random_state,
            keep_missing=keep_missing,
            **kwargs,
        )
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")
        if focus not in {"low", "medium", "high"}:
            raise ValueError("focus must be one of: low, medium, high")
        if focus_weight <= 0:
            raise ValueError("focus_weight must be greater than zero")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        self.sample_size = sample_size
        self.loss_key = loss_key
        self.focus = focus
        self.focus_weight = focus_weight
        self.quantiles = quantiles
        self.replacement = replacement
        self.temperature = temperature
        self.random_state = random_state
        self.keep_missing = keep_missing

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        records = [record for dataset in data for record in dataset]
        selected_indices = self._sample_indices(records)
        return [records[index] for index in selected_indices]

    def after_forward(
        self,
        *,
        batch: Any,
        signal: Any,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> None:
        del context, kwargs
        if signal.per_sample_losses is None:
            return
        latest_losses = state.data.setdefault("latest_losses", {})
        for index, loss in zip(batch.indices, signal.per_sample_losses, strict=True):
            latest_losses[int(index)] = float(loss)

    def on_update_boundary(
        self,
        *,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        del context
        candidates = kwargs.get("candidates")
        if candidates is None:
            return None
        losses = state.data.get("latest_losses", {})
        indexed_records = list(enumerate(candidates))
        available: list[tuple[int, float]] = []
        for index, record in indexed_records:
            record_loss = losses.get(index, self.get_field(record, self.loss_key, None))
            if record_loss is None:
                continue
            available.append((index, float(record_loss)))
        if not available:
            return None
        selected_indices = self._sample_pairs(available)
        return PolicyAction(
            selected_indices=selected_indices,
            metadata={"selector": self.name, "focus": self.focus},
        )

    def _sample_indices(self, records: Sequence[Record]) -> list[int]:
        indexed_losses: list[tuple[int, float]] = []
        kept_missing: list[int] = []
        for index, record in enumerate(records):
            record_loss = self.get_field(record, self.loss_key, None)
            if record_loss is None:
                if self.keep_missing:
                    kept_missing.append(index)
                continue
            indexed_losses.append((index, float(record_loss)))
        if not indexed_losses:
            return kept_missing[: self.sample_size]
        selected = self._sample_pairs(indexed_losses)
        if len(selected) < self.sample_size and kept_missing:
            for index in kept_missing:
                if len(selected) >= self.sample_size:
                    break
                if index not in selected:
                    selected.append(index)
        return selected[: self.sample_size]

    def _sample_pairs(self, indexed_losses: Sequence[tuple[int, float]]) -> list[int]:
        ordered_losses = sorted(loss for _, loss in indexed_losses)
        q1 = _quantile(ordered_losses, self.quantiles[0])
        q2 = _quantile(ordered_losses, self.quantiles[1])
        weights: list[float] = []
        indices: list[int] = []
        for index, loss in indexed_losses:
            weight = 1.0
            if self.focus == "low" and loss <= q1:
                weight *= self.focus_weight
            elif self.focus == "medium" and q1 < loss <= q2:
                weight *= self.focus_weight
            elif self.focus == "high" and loss > q2:
                weight *= self.focus_weight
            indices.append(index)
            weights.append(weight ** (1.0 / self.temperature))

        rng = random.Random(self.random_state)
        if self.replacement or self.sample_size > len(indices):
            return rng.choices(indices, weights=weights, k=self.sample_size)
        selected: list[int] = []
        pool_indices = list(indices)
        pool_weights = list(weights)
        for _ in range(min(self.sample_size, len(pool_indices))):
            total = sum(pool_weights)
            if total <= 0:
                choice_position = rng.randrange(len(pool_indices))
            else:
                threshold = rng.random() * total
                running = 0.0
                choice_position = len(pool_indices) - 1
                for position, weight in enumerate(pool_weights):
                    running += weight
                    if running >= threshold:
                        choice_position = position
                        break
            selected.append(pool_indices.pop(choice_position))
            pool_weights.pop(choice_position)
        return selected


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction
