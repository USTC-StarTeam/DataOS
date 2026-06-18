"""Full autonomous skeleton, all-stub (no GPU / no LLM / no claude CLI).

Stop policy is a single wall-clock condition; tests inject a fake monotonic clock so the
loop runs an exact number of rounds before the budget trips. Planner/reviewer use the
deterministic `fake` backend whose fixed-JSON reply is parsed back into PlanOut/ReviewOut.
"""

import json
from pathlib import Path

from dataos_auto.config import AutoRunConfig, Budget
from dataos_auto.loop import run_auto

PLAN = json.dumps(
    {
        "goal": "stub round",
        "experiments": [
            {"name": "exp-a", "train": {"mode": "stub"}},
            {"name": "exp-b", "train": {"mode": "stub"}},
        ],
        "rationale": "stub",
    }
)
REVIEW = json.dumps({"conclusion": "stub round done", "direction": "keep going"})


def _cfg(ws):
    return AutoRunConfig(
        project_id="p",
        workspace=ws,
        backend="stub",
        stub=True,
        gpus=[],
        poll_interval_s=0.05,
        stub_replies={"planner": PLAN, "reviewer": REVIEW},
        budget=Budget(max_wall_clock_s=1.0),
    )


def _clock_for_rounds(n):
    """started -> 0, then `n` round-checks at 0 (< budget 1.0), then a huge value to trip stop."""
    seq = iter([0.0] + [0.0] * n + [1e9] * 1000)
    return lambda: next(seq)


def test_stop_wall_clock_after_n_rounds(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(2))
    assert res["stop_reason"] == "budget:wall_clock"
    assert res["rounds_completed"] == 2
    pdir = Path(ws) / "p"
    for r in ("round-0000", "round-0001"):
        assert (pdir / r / "plan.json").is_file()
        assert (pdir / r / "review.json").is_file()
        for e in ("exp-a", "exp-b"):
            rep = json.loads((pdir / r / "experiments" / e / "report.json").read_text())
            assert rep["status"] == "completed"
    # prepared data + agent workspaces live OUTSIDE the project dir (not mis-counted as studies)
    assert (Path(ws) / "_prepared" / "p").exists()
    assert res["summary"]["n_studies"] == 2


def test_stop_wall_clock_immediate(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(0))
    assert res["stop_reason"] == "budget:wall_clock"
    assert res["rounds_completed"] == 0


def test_best_over_history_final_model(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(2))
    fm = res["final_model"]
    assert fm and fm["source"] == "best_over_history" and fm["score"] == 0.5
    assert (Path(ws) / "p" / "final_model.json").is_file()


def test_resume_skips_done_round(tmp_path):
    ws = str(tmp_path / "ws")
    run_auto(_cfg(ws), _clock=_clock_for_rounds(1))  # round-0000 only
    rp = Path(ws) / "p" / "round-0000" / "experiments" / "exp-a" / "report.json"
    before = json.loads(rp.read_text())

    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(1))  # resume: runs round-0001
    after = json.loads(rp.read_text())
    assert after["id"] == before["id"]  # round-0000 NOT re-run on resume
    assert res["rounds_completed"] == 2
    assert (Path(ws) / "p" / "round-0001" / "review.json").is_file()
