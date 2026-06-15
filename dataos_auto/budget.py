"""The three hard stop conditions — pure functions, no LLM, no IO.

This is the runtime enforcement that the skill's prose budget never had: the loop
stops on exactly one of three conditions, first to fire wins, and the LLM never
participates in the decision.

    1. criterion        — bench score target met
    2. budget:rounds    — max rounds reached
    3. budget:wall_clock — max wall-clock seconds elapsed
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


def stop_reason(
    *,
    round_idx: int,
    started_at: float,
    best: dict | None,
    criterion: Criterion,
    budget: Budget,
    now: float | None = None,
) -> str | None:
    """Return the stop reason, or None to continue. Checked at the TOP of each round.

    ``round_idx`` is the 0-based count of rounds already completed, so
    ``round_idx >= max_rounds`` stops *after* that many rounds have run.
    ``now`` is injectable for testing; defaults to ``time.monotonic()``.
    """
    if criterion_met(best, criterion):
        return "criterion"
    if budget.max_rounds is not None and round_idx >= budget.max_rounds:
        return "budget:rounds"
    if budget.max_wall_clock_s is not None:
        elapsed = (now if now is not None else time.monotonic()) - started_at
        if elapsed >= budget.max_wall_clock_s:
            return "budget:wall_clock"
    return None
