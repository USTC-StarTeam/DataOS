"""The three hard stop conditions — pure functions, fully deterministic (inject `now`)."""

from dataos_auto.budget import criterion_met, stop_reason
from dataos_auto.config import Budget, Criterion


def test_criterion_met_score():
    assert criterion_met({"score": 0.5}, Criterion(threshold=0.4))
    assert not criterion_met({"score": 0.3}, Criterion(threshold=0.4))
    assert not criterion_met(None, Criterion(threshold=0.4))  # no completed eval → never met
    assert not criterion_met({"score": 0.5}, Criterion(threshold=None))  # no target → never
    assert criterion_met({"score": 0.3}, Criterion(threshold=0.4, direction="min"))


def test_criterion_met_named_bench():
    best = {"score": 0.5, "per_task": {"gsm8k": 0.9}}
    assert criterion_met(best, Criterion(metric="gsm8k", threshold=0.8))
    assert not criterion_met(best, Criterion(metric="gsm8k", threshold=0.95))
    assert not criterion_met({"score": 0.5}, Criterion(metric="gsm8k", threshold=0.1))  # no per_task


def test_stop_reason_criterion_wins_over_budget():
    assert (
        stop_reason(
            round_idx=5,
            started_at=0.0,
            best={"score": 0.9},
            criterion=Criterion(threshold=0.5),
            budget=Budget(max_rounds=1),
            now=999.0,
        )
        == "criterion"
    )


def test_stop_reason_rounds():
    common = dict(started_at=0.0, best=None, criterion=Criterion(threshold=None), now=1.0)
    assert stop_reason(round_idx=2, budget=Budget(max_rounds=2), **common) == "budget:rounds"
    assert stop_reason(round_idx=1, budget=Budget(max_rounds=2), **common) is None


def test_stop_reason_wall_clock():
    common = dict(round_idx=0, best=None, criterion=Criterion(threshold=None))
    assert (
        stop_reason(started_at=100.0, budget=Budget(max_wall_clock_s=10.0), now=111.0, **common)
        == "budget:wall_clock"
    )
    assert (
        stop_reason(started_at=100.0, budget=Budget(max_wall_clock_s=10.0), now=105.0, **common)
        is None
    )


def test_stop_reason_none_under_all_caps():
    assert (
        stop_reason(
            round_idx=0,
            started_at=0.0,
            best={"score": 0.1},
            criterion=Criterion(threshold=0.9),
            budget=Budget(max_rounds=3, max_wall_clock_s=1000.0),
            now=1.0,
        )
        is None
    )
