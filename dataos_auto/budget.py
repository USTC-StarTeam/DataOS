"""The PTB stop policy — a pure function, no LLM, no IO.

On agent-version-ptb the PostTrainBench protocol is a SINGLE stop condition: a wall-clock
budget (10h / 1 GPU per (base, bench)). The run uses the full budget and submits the
best-over-history model — neither a score target nor a round count stops it.
``criterion_met`` is kept only as a reporting helper ("did the target get hit?"); it does
NOT drive stopping.
"""

from __future__ import annotations

import time

from .config import Budget, Criterion


def _best_metric(best: dict | None, metric: str) -> float | None:
    if not best:
        return None
    if metric in (None, "", "score"):
        v = best.get("score")
    else:  # named bench → per-task score on the best experiment (best-effort)
        v = (best.get("per_task") or {}).get(metric)
    return float(v) if isinstance(v, (int, float)) else None


def criterion_met(best: dict | None, c: Criterion) -> bool:
    """True iff the best experiment's score meets the target.

    ``best is None`` (no completed eval yet) → never met."""
    if c is None or c.threshold is None:
        return False
    score = _best_metric(best, c.metric)
    if score is None:
        return False
    return score >= c.threshold if c.direction == "max" else score <= c.threshold


def stop_reason(*, started_at: float, budget: Budget, now: float | None = None) -> str | None:
    """The single PTB stop condition: return "budget:wall_clock" once the wall-clock budget
    is spent, else None. ``now`` is injectable for testing (defaults to ``time.monotonic()``)."""
    if budget.max_wall_clock_s is None:
        return None
    elapsed = (now if now is not None else time.monotonic()) - started_at
    return "budget:wall_clock" if elapsed >= budget.max_wall_clock_s else None
