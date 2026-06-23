"""Full agentic loop, all-stub (no GPU / no LLM / no claude CLI)."""

import json
from pathlib import Path

from dataos.agent.config import AutoRunConfig, Budget
from dataos.agent.loop import run_auto


def _plan(parent: str, name: str) -> str:
    return json.dumps(
        {
            "parent_node_id": parent,
            "name": name,
            "hypothesis": f"grow {name}",
            "estimated_hours": 0.1,
            "data": {},
            "train": {"mode": "stub"},
            "eval_plan": "stub eval",
            "expected_signal": "completed stub score",
            "subagent_brief": f"run {name}",
            "rationale": "stub",
        }
    )


REVIEW = json.dumps(
    {
        "interpretation": "stub node done",
        "lesson": "stub lesson",
        "recommendation_for_future_growth": "continue from the best node",
    }
)


def _cfg(ws):
    return AutoRunConfig(
        project_id="p",
        workspace=ws,
        backend="stub",
        stub=True,
        gpus=[],
        poll_interval_s=0.05,
        stub_replies={
            "plan": [_plan("n0000", "child-0"), _plan("n0001", "child-1")],
            "review": REVIEW,
        },
        budget=Budget(max_wall_clock_s=1.0),
    )


def _clock_for_rounds(n):
    """started -> 0, then `n` in-budget loop checks, then a huge value to trip stop."""
    seq = iter([0.0] + [0.0] * n + [1e9] * 1000)
    return lambda: next(seq)


def test_stop_wall_clock_after_n_rounds(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(2))
    assert res["stop_reason"] == "budget:wall_clock"
    assert res["rounds_completed"] == 2

    pdir = Path(ws) / "p"
    for r, exp in {"round-0000": "child-0", "round-0001": "child-1"}.items():
        assert (pdir / r / "plan.json").is_file()
        assert (pdir / r / "review.json").is_file()
        rep = json.loads((pdir / r / "experiments" / exp / "report.json").read_text())
        assert rep["status"] == "completed"

    assert (Path(ws) / "_prepared" / "p").exists()
    assert res["summary"]["n_studies"] == 2

    tree = json.loads((pdir / "tree.json").read_text())
    assert len(tree["nodes"]) == 3
    assert tree["nodes"][1]["parent"] == "n0000"
    assert tree["nodes"][2]["parent"] == "n0001"
    assert tree["nodes"][2]["init_model_path"] == tree["nodes"][1]["model_path"]
    scored = [n for n in tree["nodes"] if n["score"] is not None]
    assert len(scored) == 2 and all(n["score"] == 0.5 for n in scored)


def test_stop_wall_clock_immediate(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(0))
    assert res["stop_reason"] == "budget:wall_clock"
    assert res["rounds_completed"] == 0


def test_best_over_tree_final_model(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(2))
    fm = res["final_model"]
    assert fm and fm["source"] == "best_over_tree" and fm["score"] == 0.5
    assert (Path(ws) / "p" / "final_model.json").is_file()


def test_resume_skips_done_round(tmp_path):
    ws = str(tmp_path / "ws")
    run_auto(_cfg(ws), _clock=_clock_for_rounds(1))
    rp = Path(ws) / "p" / "round-0000" / "experiments" / "child-0" / "report.json"
    before = json.loads(rp.read_text())

    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(1))
    after = json.loads(rp.read_text())
    assert after["id"] == before["id"]
    assert res["rounds_completed"] == 2
    assert (Path(ws) / "p" / "round-0001" / "review.json").is_file()
