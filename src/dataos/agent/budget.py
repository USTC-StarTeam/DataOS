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


#: Ceiling on the queue-wait exemption, as a fraction of the budget. Without it a congested
#: cluster turns a 10h run into an open-ended one — the 2026-08-11 smol3 batch already ran
#: gsm8k for 11.5h on a 10h budget.
_MAX_CREDIT_FRACTION = 0.5


def credited_deadline(*, started_at: float, budget: Budget, queue_credit_s: float = 0.0) -> float:
    """The one deadline the whole run uses (monotonic seconds).

    The budget tracks IDEAL run time, so Slurm queue-wait is exempt — cluster contention is
    not experiment work — but the exemption is capped at :data:`_MAX_CREDIT_FRACTION` of the
    budget so real wall-clock stays bounded.

    ONE function because the run previously computed this twice with different rules: the
    loop's stop check subtracted the credit while ``fan_out``'s experiment deadline did not.
    That opened a window — real elapsed past ``started+budget`` but credited elapsed still
    under it — where no experiment could ever launch and the loop would never stop. The run
    then span in place: plan -> fan_out skips -> review an experiment that never ran -> plan.
    77 of the smol3 batch's 153 rounds were such zombies (gsm8k 36, healthbench 28,
    aime2025 12), each burning a leader planning call plus a review and flooding
    ``recent_reviews`` with "The prediction cannot be evaluated".
    """
    if budget.max_wall_clock_s is None:
        return float("inf")
    credit = min(max(queue_credit_s, 0.0), _MAX_CREDIT_FRACTION * budget.max_wall_clock_s)
    return started_at + budget.max_wall_clock_s + credit


def stop_reason(*, started_at: float, budget: Budget, now: float | None = None,
                queue_credit_s: float = 0.0) -> str | None:
    """The single PTB stop condition: return "budget:wall_clock" once the budget is spent,
    else None. Shares :func:`credited_deadline` with the experiment launcher so the two can
    never disagree. ``now`` is injectable for testing (defaults to ``time.monotonic()``)."""
    if budget.max_wall_clock_s is None:
        return None
    deadline = credited_deadline(
        started_at=started_at, budget=budget, queue_credit_s=queue_credit_s
    )
    now = now if now is not None else time.monotonic()
    return "budget:wall_clock" if now >= deadline else None
