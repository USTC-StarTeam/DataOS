"""PTB stop policy: a single wall-clock condition (pure, deterministic via injected `now`)."""

from dataos_auto.budget import criterion_met, stop_reason
from dataos_auto.config import PTB_WALL_CLOCK_S, Budget, Criterion


def test_wall_clock_is_the_only_stop():
    b = Budget(max_wall_clock_s=10.0)
    assert stop_reason(started_at=100.0, budget=b, now=111.0) == "budget:wall_clock"
    assert stop_reason(started_at=100.0, budget=b, now=105.0) is None


def test_no_wall_clock_budget_never_stops():
    assert stop_reason(started_at=0.0, budget=Budget(max_wall_clock_s=None), now=1e9) is None


def test_rounds_are_not_a_stop_condition():
    # max_rounds set but no wall-clock budget → never stops (rounds is kept for record only)
    b = Budget(max_rounds=1, max_wall_clock_s=None)
    assert stop_reason(started_at=0.0, budget=b, now=1e9) is None


def test_default_budget_is_10h():
    assert PTB_WALL_CLOCK_S == 36000
    assert Budget(max_wall_clock_s=PTB_WALL_CLOCK_S).max_wall_clock_s == 36000


def test_criterion_met_is_only_a_reporting_helper():
    # kept to report "did the target get hit?" — NOT used by stop_reason
    assert criterion_met({"score": 0.5}, Criterion(threshold=0.4))
    assert not criterion_met(None, Criterion(threshold=0.4))
    assert not criterion_met({"score": 0.5}, Criterion(threshold=None))
    assert criterion_met({"score": 0.3}, Criterion(threshold=0.4, direction="min"))
