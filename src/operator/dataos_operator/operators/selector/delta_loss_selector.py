"""Delta-loss selector that tracks how sample difficulty changes over updates."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Dataset, Record, SelectorOperator
from dataos_operator.runtime import OperatorState, PolicyAction, RuntimeContext


class DeltaLossSelector(SelectorOperator):
    """Select records from a moving delta-loss window over training progress.

    Implementation: per record compute the delta-loss ``initial_loss - current_loss`` (read from
    ``initial_loss_key``/``loss_key`` offline, or accumulated across steps via the runtime
    ``after_forward`` hook), sort by delta descending, then slide a window of size
    ``window_fraction`` of the pool whose start advances with training ``progress``
    (``step/total_updates``); in-window records with positive delta get weight 1.0 and the rest
    ``floor_weight``, then weighted-sample ``sample_size`` without replacement (seeded by
    ``random_state``).
    Dependency: pure-rule operator, no external dependencies; online use requires per-sample
    losses fed through the PolicyRuntime, offline ``.run()`` reads them from record metadata.
    """

    def __init__(
        self,
        sample_size: int,
        *,
        initial_loss_key: str = "metadata.runtime.initial_loss",
        loss_key: str = "metadata.runtime.loss",
        window_fraction: float = 0.2,
        floor_weight: float = 0.025,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_size=sample_size,
            initial_loss_key=initial_loss_key,
            loss_key=loss_key,
            window_fraction=window_fraction,
            floor_weight=floor_weight,
            random_state=random_state,
            **kwargs,
        )
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")
        if not 0 < window_fraction <= 1:
            raise ValueError("window_fraction must be in (0, 1]")
        if floor_weight < 0:
            raise ValueError("floor_weight must be non-negative")

        self.sample_size = sample_size
        self.initial_loss_key = initial_loss_key
        self.loss_key = loss_key
        self.window_fraction = window_fraction
        self.floor_weight = floor_weight
        self.random_state = random_state

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        records = [record for dataset in data for record in dataset]
        selected_indices = self._sample_indices(records, progress=0.5)
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
        initial_losses = state.data.setdefault("initial_losses", {})
        for index, loss in zip(batch.indices, signal.per_sample_losses, strict=True):
            latest_losses[int(index)] = float(loss)
            initial_losses.setdefault(int(index), float(loss))

    def on_update_boundary(
        self,
        *,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        candidates = kwargs.get("candidates")
        if candidates is None:
            return None
        total_updates = max(int(kwargs.get("total_updates", context.metadata.get("total_updates", 1))), 1)
        progress = min(max(context.step / total_updates, 0.0), 1.0)
        selected_indices = self._sample_with_state(candidates, state, progress)
        return PolicyAction(
            selected_indices=selected_indices,
            metadata={"selector": self.name, "progress": progress},
        )

    def _sample_indices(self, records: Sequence[Record], *, progress: float) -> list[int]:
        pairs: list[tuple[int, float]] = []
        for index, record in enumerate(records):
            initial_loss = self.get_field(record, self.initial_loss_key, None)
            current_loss = self.get_field(record, self.loss_key, None)
            if initial_loss is None or current_loss is None:
                continue
            pairs.append((index, float(initial_loss) - float(current_loss)))
        if not pairs:
            return []
        return self._windowed_sample(pairs, progress=progress)

    def _sample_with_state(
        self,
        records: Sequence[Record],
        state: OperatorState,
        progress: float,
    ) -> list[int]:
        latest_losses = state.data.get("latest_losses", {})
        initial_losses = state.data.get("initial_losses", {})
        pairs: list[tuple[int, float]] = []
        for index, record in enumerate(records):
            current_loss = latest_losses.get(index, self.get_field(record, self.loss_key, None))
            initial_loss = initial_losses.get(index, self.get_field(record, self.initial_loss_key, None))
            if current_loss is None:
                continue
            if initial_loss is None:
                initial_loss = current_loss
            pairs.append((index, float(initial_loss) - float(current_loss)))
        if not pairs:
            return []
        return self._windowed_sample(pairs, progress=progress)

    def _windowed_sample(self, pairs: Sequence[tuple[int, float]], *, progress: float) -> list[int]:
        ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
        window_size = max(1, int(len(ordered) * self.window_fraction))
        max_start = max(len(ordered) - window_size, 0)
        window_start = int(progress * max_start)
        window_end = min(window_start + window_size, len(ordered))
        rng = random.Random(self.random_state)

        weights = [self.floor_weight for _ in ordered]
        for position in range(window_start, window_end):
            if ordered[position][1] > 0:
                weights[position] = 1.0

        indices = [index for index, _ in ordered]
        if self.sample_size >= len(indices):
            return indices

        selected: list[int] = []
        pool_indices = list(indices)
        pool_weights = list(weights)
        for _ in range(self.sample_size):
            total = sum(pool_weights)
            threshold = rng.random() * total if total > 0 else 0.0
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
