"""Reference online operator: hardness-based per-sample loss reweighting.

Torch-free numerical core; the backend's trainer glue feeds live per-sample losses in via
the ``reweight`` batch payload (see :mod:`dataos_train.online` for the protocol).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .registry import register_online_op


def reweight_factors(
    losses: Sequence[float], alpha: float = 1.0, clip: float = 10.0, eps: float = 1e-8
) -> list[float]:
    """Hardness reweighting: factor_i ∝ (loss_i / mean_loss)^alpha, normalized to mean 1.

    alpha=0 -> uniform (all 1.0); larger alpha -> more weight on hard (high-loss) samples.
    """
    n = len(losses)
    if n == 0:
        return []
    mean = sum(losses) / n
    if mean <= eps or alpha == 0.0:
        return [1.0] * n
    raw = [min((max(loss, 0.0) / mean) ** alpha, clip) for loss in losses]
    m = sum(raw) / n
    if m <= eps:
        return [1.0] * n
    return [r / m for r in raw]


class LossReweightHook:
    """Reference online operator: hardness-based per-sample loss reweighting.

    Up-weights high-loss (hard) samples within the current batch — an online, data-aware
    signal derived from the *live* model state each step. Demonstrates the ``compute_loss``
    patch path that distinguishes online ops from offline preprocessing.
    """

    name = "loss_reweight"

    def __init__(self, hook: str | None = None, alpha: float = 1.0, clip: float = 10.0) -> None:
        hook_point = hook or "reweight"
        if hook_point != "reweight":
            raise ValueError(f"{self.name} supports hook point 'reweight' only, got {hook!r}")
        self.hook_point = hook_point
        self.alpha = float(alpha)
        self.clip = float(clip)

    def setup(self, ctx: Any) -> None:  # stateless
        return None

    def __call__(self, batch: dict) -> list[float]:
        return reweight_factors(batch["losses"], alpha=self.alpha, clip=self.clip)


register_online_op("loss_reweight", LossReweightHook)


class EpochSubsetSelectHook:
    """Reference ``select`` operator: keep only a leading fraction of rows from epoch 2 on.

    Demonstrates the epoch-boundary decision path end to end (and doubles as its smoke
    test): epoch 1 trains on the full dataset (pass-through), ``decide`` then narrows every
    later epoch to the first ``fraction`` of rows.
    """

    name = "epoch_subset_select"

    def __init__(self, hook: str | None = None, fraction: float = 0.5) -> None:
        hook_point = hook or "select"
        if hook_point != "select":
            raise ValueError(f"{self.name} supports hook point 'select' only, got {hook!r}")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.hook_point = hook_point
        self.fraction = float(fraction)

    def setup(self, ctx: Any) -> None:  # stateless
        return None

    def observe(self, batch: dict) -> None:  # decision needs no per-row stats
        return None

    def decide(self, ctx: dict) -> dict | None:
        n = ctx.get("n_rows")
        if not n:
            return None  # dataset size unknown — leave the sampler untouched
        return {"indices": list(range(max(1, int(n * self.fraction)))), "domain_weights": None}


register_online_op("epoch_subset_select", EpochSubsetSelectHook)


def apply_reweight(batch: dict, hooks: Sequence[Any]) -> list[float]:
    """Chain all ``reweight`` hooks into combined per-sample factors (≈ mean 1)."""
    factors = [1.0] * len(batch["losses"])
    for hook in hooks:
        if getattr(hook, "hook_point", "") != "reweight":
            continue
        partial = hook(batch)
        factors = [f * g for f, g in zip(factors, partial, strict=False)]
    return factors
