"""DoReMi-style domain mixer driven by excess loss over a reference model."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from dataos_operator import Dataset
from dataos_operator.core import MixerOperator, Record
from dataos_operator.runtime import ArtifactRef, OperatorState, PolicyAction, RuntimeContext


class DoremiMixer(MixerOperator):
    """Approximate DoReMi with online excess-loss domain reweighting.

    Implementation: group records by ``domain_key``; the offline ``.run()`` path mixes with uniform
    domain weights. In the runtime, ``after_forward`` accumulates each domain's excess loss
    (``current_loss - reference_loss``, floored at 0); each ``on_update_boundary`` updates domain
    weights multiplicatively by ``exp(eta * (score - mean_score))`` then smooths toward uniform by
    ``epsilon``, appending to a running history; ``finalize`` returns the time-averaged weights.
    ``mix_with_weights`` resamples each domain to its quota for the target size (seeded by
    ``random_state``).
    Dependency: pure-rule operator, no external dependencies; the online path needs per-sample
    losses and reference losses fed through the PolicyRuntime.
    """

    def __init__(
        self,
        *,
        domain_key: str = "metadata.source",
        reference_loss_key: str = "metadata.runtime.reference_loss",
        eta: float = 1.0,
        epsilon: float = 1e-3,
        output_size: int | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            domain_key=domain_key,
            reference_loss_key=reference_loss_key,
            eta=eta,
            epsilon=epsilon,
            output_size=output_size,
            random_state=random_state,
            **kwargs,
        )
        if eta <= 0:
            raise ValueError("eta must be greater than zero")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        self.domain_key = domain_key
        self.reference_loss_key = reference_loss_key
        self.eta = eta
        self.epsilon = epsilon
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
            metadata={"doremi": {"domain_weights": domain_weights}},
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
        del context
        reference_losses = kwargs.get("reference_losses")
        if signal.per_sample_losses is None or batch.domain_ids is None or reference_losses is None:
            return
        domain_stats = state.data.setdefault("domain_stats", {})
        for domain_id, current_loss, reference_loss in zip(
            batch.domain_ids,
            signal.per_sample_losses,
            reference_losses,
            strict=True,
        ):
            stats = domain_stats.setdefault(str(domain_id), {"sum": 0.0, "count": 0})
            excess = max(float(current_loss) - float(reference_loss), 0.0)
            stats["sum"] += excess
            stats["count"] += 1

    def on_update_boundary(
        self,
        *,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        del context, kwargs
        domain_stats = state.data.get("domain_stats")
        if not domain_stats:
            return None
        previous_weights = state.data.get("domain_weights")
        if previous_weights is None:
            previous_weights = _uniform_weights({domain: [] for domain in domain_stats})

        scores = {
            domain: stats["sum"] / max(stats["count"], 1)
            for domain, stats in domain_stats.items()
        }
        mean_score = sum(scores.values()) / max(len(scores), 1)
        alpha_prime = {
            domain: previous_weights.get(domain, 0.0) * math.exp(self.eta * (score - mean_score))
            for domain, score in scores.items()
        }
        total = sum(alpha_prime.values())
        if total <= 0:
            return None
        uniform = 1.0 / len(alpha_prime)
        updated = {
            domain: (1 - self.epsilon) * (value / total) + self.epsilon * uniform
            for domain, value in alpha_prime.items()
        }
        total_updated = sum(updated.values())
        updated = {domain: value / total_updated for domain, value in updated.items()}
        state.data["domain_weights"] = updated
        state.data.setdefault("history", []).append(updated)
        state.data["domain_stats"] = {}
        return PolicyAction(
            domain_weights=updated,
            metadata={"mixer": self.name, "scores": scores},
            artifacts=[
                ArtifactRef(
                    name="doremi_average_weights",
                    kind="domain_weights",
                    metadata={"average_weights": _average_weights(state.data["history"])},
                )
            ],
        )

    def finalize(
        self,
        *,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        del context, kwargs
        history = state.data.get("history", [])
        if not history:
            return None
        average_weights = _average_weights(history)
        return PolicyAction(
            domain_weights=average_weights,
            artifacts=[
                ArtifactRef(
                    name="doremi_average_weights_final",
                    kind="domain_weights",
                    metadata={"average_weights": average_weights},
                )
            ],
            metadata={"mixer": self.name, "stage": "finalize"},
        )


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


def _average_weights(history: Sequence[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    if not history:
        return {}
    for weights in history:
        for domain, value in weights.items():
            totals[domain] += value
    return {domain: value / len(history) for domain, value in totals.items()}
