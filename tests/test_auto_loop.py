"""Full autonomous skeleton, all-stub (no GPU / no LLM / no claude CLI).

Verifies the three stop reasons and crash-safe project-scoped resume. The planner and
reviewer use the deterministic `fake` backend whose fixed-JSON reply is parsed back into
PlanOut / ReviewOut by the facade's `output=` mechanism.
"""

import json
from pathlib import Path

from dataos_auto.config import AutoRunConfig, Budget, Criterion
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


def _cfg(ws, *, criterion=None, budget=None):
    return AutoRunConfig(
        project_id="p",
        workspace=ws,
        backend="stub",
        stub=True,
        gpus=[],
        poll_interval_s=0.05,
        stub_replies={"planner": PLAN, "reviewer": REVIEW},
        criterion=criterion or Criterion(),
        budget=budget or Budget(),
    )


def test_stop_budget_rounds(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws, budget=Budget(max_rounds=2)))
    assert res["stop_reason"] == "budget:rounds"
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


def test_stop_criterion(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws, criterion=Criterion(threshold=0.4), budget=Budget(max_rounds=9)))
    assert res["stop_reason"] == "criterion"
    assert res["rounds_completed"] == 1
    assert res["best"]["score"] == 0.5


def test_stop_wall_clock_immediate(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws, budget=Budget(max_wall_clock_s=0.0, max_rounds=9)))
    assert res["stop_reason"] == "budget:wall_clock"
    assert res["rounds_completed"] == 0


def test_resume_skips_done_round(tmp_path):
    ws = str(tmp_path / "ws")
    run_auto(_cfg(ws, budget=Budget(max_rounds=1)))
    rp = Path(ws) / "p" / "round-0000" / "experiments" / "exp-a" / "report.json"
    before = json.loads(rp.read_text())

    res = run_auto(_cfg(ws, budget=Budget(max_rounds=2)))
    assert res["rounds_completed"] == 2
    after = json.loads(rp.read_text())
    assert after["id"] == before["id"]  # round-0000 NOT re-run on resume
    assert (Path(ws) / "p" / "round-0001" / "review.json").is_file()
