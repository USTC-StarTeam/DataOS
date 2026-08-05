"""ODM-style online domain mixer using an Exp3 moving-average reward estimate."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from dataos_operator import Dataset
from dataos_operator.core import MixerOperator, Record
from dataos_operator.runtime import OperatorState, PolicyAction, RuntimeContext


class OdmMixer(MixerOperator):
    """Online data mixing with Exp3-style exploration and moving-average rewards.

    Implementation: group records by ``domain_key``; the offline ``.run()`` path mixes uniformly.
    In the runtime, ``after_forward`` records each domain's mean batch loss as its reward; after
    ``warmup_steps`` each ``on_update_boundary`` runs Exp3—importance-correct the reward by the
    domain's current sampling probability, update a moving-average estimate (decay ``alpha``, gain
    ``reward_scale``), then form new domain weights via a softmax mixed with a decaying exploration
    floor (at least ``min_exploration_rate``). ``mix_with_weights`` resamples each domain to its
    quota for the target size (seeded by ``random_state``).
    Dependency: pure-rule operator, no external dependencies; the online path needs per-sample
    losses plus batch domain ids fed through the PolicyRuntime.
    """

    def __init__(
        self,
        *,
        domain_key: str = "metadata.source",
        alpha: float = 0.9,
        warmup_steps: int = 0,
        reward_scale: float = 15.0,
        min_exploration_rate: float = 0.03,
        output_size: int | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            domain_key=domain_key,
            alpha=alpha,
            warmup_steps=warmup_steps,
            reward_scale=reward_scale,
            min_exploration_rate=min_exploration_rate,
            output_size=output_size,
            random_state=random_state,
            **kwargs,
        )
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1)")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

        self.domain_key = domain_key
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.reward_scale = reward_scale
        self.min_exploration_rate = min_exploration_rate
        self.output_size = output_size
        self.random_state = random_state

    def mix(self, data: Sequence[Dataset]) -> Dataset:
        records = [record for dataset in data for record in dataset]
        grouped: dict[str, list[Record]] = defaultdict(list)
        for record in records:
            grouped[str(self.require_field(record, self.domain_key))].append(record)
        weights = _uniform_weights(grouped)
        return self.mix_with_weights(data, domain_weights=weights)

    def mix_with_weights(
        self,
        data: Sequence[Dataset],
        *,
        domain_weights: dict[str, float],
        output_size: int | None = None,
    ) -> Dataset:
        records = [record for dataset in data for record in dataset]
        grouped: dict[str, list[Record]] = defaultdict(list)
        for record in records:
            grouped[str(self.require_field(record, self.domain_key))].append(record)
        return Dataset.from_records(
            _sample_grouped_records(
                grouped,
                domain_weights,
                output_size or self.output_size or len(records),
                self.random_state,
            ),
            metadata={"odm": {"domain_weights": domain_weights}},
        )

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
        if signal.per_sample_losses is None or batch.domain_ids is None or not batch.domain_ids:
            return
        counts: dict[str, int] = defaultdict(int)
        losses: dict[str, float] = defaultdict(float)
        for domain_id, loss in zip(batch.domain_ids, signal.per_sample_losses, strict=True):
            counts[str(domain_id)] += 1
            losses[str(domain_id)] += float(loss)
        batch_rewards = state.data.setdefault("batch_rewards", {})
        for domain, count in counts.items():
            batch_rewards[domain] = losses[domain] / count

    def on_update_boundary(
        self,
        *,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        del kwargs
        batch_rewards = state.data.get("batch_rewards")
        domain_weights = state.data.get("domain_weights", {})
        estimated_rewards = state.data.get("estimated_rewards", {})
        if not batch_rewards:
            return None
        if not domain_weights:
            domain_weights = _uniform_weights({domain: [] for domain in batch_rewards})
        if not estimated_rewards:
            estimated_rewards = {domain: 0.0 for domain in domain_weights}

        if context.step > self.warmup_steps:
            for domain, batch_loss in batch_rewards.items():
                probability = max(domain_weights.get(domain, 1e-8), 1e-8)
                reward = batch_loss * self.reward_scale
                estimated_rewards[domain] = (
                    self.alpha * estimated_rewards.get(domain, 0.0)
                    + (1 - self.alpha) * (reward / probability)
                )

            exploration_rate = self._exploration_rate(max(context.step - self.warmup_steps, 1), len(domain_weights))
            shifted = [exploration_rate * estimated_rewards[domain] for domain in sorted(domain_weights)]
            max_value = max(shifted)
            exp_values = [math.exp(max(value - max_value, -5.0)) for value in shifted]
            total = sum(exp_values)
            updated: dict[str, float] = {}
            for domain, exp_value in zip(sorted(domain_weights), exp_values, strict=True):
                updated[domain] = exp_value / total * max(0.0, 1 - len(domain_weights) * exploration_rate) + exploration_rate
            total_updated = sum(updated.values())
            domain_weights = {domain: value / total_updated for domain, value in updated.items()}
            state.data["exploration_rate"] = exploration_rate

        state.data["batch_rewards"] = {}
        state.data["estimated_rewards"] = estimated_rewards
        state.data["domain_weights"] = domain_weights
        return PolicyAction(
            domain_weights=domain_weights,
            metadata={
                "mixer": self.name,
                "estimated_rewards": estimated_rewards,
                "exploration_rate": state.data.get("exploration_rate", 1.0 / max(len(domain_weights), 1)),
            },
        )

    def _exploration_rate(self, step: int, domain_count: int) -> float:
        if step <= 0 or domain_count <= 0:
            return 1.0
        decay = math.sqrt(math.log(domain_count) / (domain_count * step))
        return max(self.min_exploration_rate, min(1.0 / domain_count, decay))


def _sample_grouped_records(
    grouped: dict[str, list[Record]],
    weights: dict[str, float],
    target_size: int,
    random_state: int,
) -> list[Record]:
    rng = random.Random(random_state)
    domains = sorted(grouped)
    selected: list[Record] = []
    for domain in domains:
        quota = int(round(target_size * weights.get(domain, 0.0)))
        records = list(grouped[domain])
        if not records:
            continue
        if quota <= len(records):
            selected.extend(rng.sample(records, quota))
        else:
            selected.extend(records)
            selected.extend(rng.choices(records, k=quota - len(records)))
    rng.shuffle(selected)
    return selected[:target_size]


def _uniform_weights(grouped: dict[str, list[Record]]) -> dict[str, float]:
    if not grouped:
        return {}
    weight = 1.0 / len(grouped)
    return {domain: weight for domain in grouped}
