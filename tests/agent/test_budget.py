"""Task-selected stop policies (pure, deterministic via injected inputs)."""

from dataos.agent import budget
from dataos.agent.budget import criterion_met, stop_reason
from dataos.agent.config import PTB_WALL_CLOCK_S, Budget, Criterion


def test_wall_clock_is_the_only_stop():
    b = Budget(max_wall_clock_s=10.0)
    assert stop_reason(started_at=100.0, budget=b, now=111.0) == "budget:wall_clock"
    assert stop_reason(started_at=100.0, budget=b, now=105.0) is None


def test_no_wall_clock_budget_never_stops():
    assert stop_reason(started_at=0.0, budget=Budget(max_wall_clock_s=None), now=1e9) is None


def test_iteration_budget_stops_at_the_exact_completed_round_count():
    b = Budget(max_rounds=1, max_wall_clock_s=None)
    assert stop_reason(started_at=0.0, budget=b, now=1e9, rounds_completed=0) is None
    assert (
        stop_reason(started_at=0.0, budget=b, now=1e9, rounds_completed=1)
        == "budget:iterations"
    )


def test_default_budget_is_10h():
    assert PTB_WALL_CLOCK_S == 36000
    assert Budget(max_wall_clock_s=PTB_WALL_CLOCK_S).max_wall_clock_s == 36000


def test_criterion_met_is_only_a_reporting_helper():
    # kept to report "did the target get hit?" — NOT used by stop_reason
    assert criterion_met({"score": 0.5}, Criterion(threshold=0.4))
    assert not criterion_met(None, Criterion(threshold=0.4))
    assert not criterion_met({"score": 0.5}, Criterion(threshold=None))
    assert criterion_met({"score": 0.3}, Criterion(threshold=0.4, direction="min"))


# --- the one deadline (queue credit, capped) ------------------------------------------------
# The run used to compute this twice with different rules: the stop check subtracted the Slurm
# queue credit, fan_out's experiment deadline did not. In the window between them nothing could
# launch and nothing would stop — 77 of the smol3 batch's 153 rounds span there.


def test_credit_extends_the_deadline_for_both_users():
    b = Budget(max_wall_clock_s=100.0)
    assert budget.credited_deadline(started_at=0.0, budget=b, queue_credit_s=30.0) == 130.0
    # the stop check reads the same line: at 129s still running, at 130s stopped
    assert budget.stop_reason(started_at=0.0, budget=b, now=129.0, queue_credit_s=30.0) is None
    assert budget.stop_reason(started_at=0.0, budget=b, now=130.0, queue_credit_s=30.0)


def test_credit_is_capped_so_a_congested_cluster_cannot_run_forever():
    b = Budget(max_wall_clock_s=100.0)
    # 500s of queue wait would otherwise push the deadline to 600s
    assert budget.credited_deadline(started_at=0.0, budget=b, queue_credit_s=500.0) == 150.0
    assert budget.stop_reason(started_at=0.0, budget=b, now=151.0, queue_credit_s=500.0)


def test_no_credit_means_the_plain_budget():
    b = Budget(max_wall_clock_s=100.0)
    assert budget.credited_deadline(started_at=10.0, budget=b) == 110.0
    assert budget.credited_deadline(started_at=0.0, budget=b, queue_credit_s=-5.0) == 100.0


def test_no_wall_clock_budget_has_no_deadline():
    assert budget.credited_deadline(started_at=0.0, budget=Budget(max_wall_clock_s=None)) == float(
        "inf"
    )
