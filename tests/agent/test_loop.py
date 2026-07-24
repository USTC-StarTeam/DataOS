"""Full agentic loop, all-stub (no GPU / no LLM / no claude CLI).

The loop's only executor is the sub-agent (``dataos.agent.subagent_executor``); here we
replace its ``fan_out`` with a fake that just writes a completed ``report.json`` per spec, so
the test exercises the loop's orchestration (plan → execute → review → tree → budget stop →
resume) without a real sub-agent / H100.
"""

import json
from pathlib import Path

import pytest

from dataos.agent.config import AutoRunConfig, Budget
from dataos.agent.loop import run_auto
from dataos.agent import subagent_executor


def _plan(parent: str, name: str) -> str:
    return json.dumps(
        {
            "parent_node_id": parent,
            "name": name,
            "hypothesis": f"grow {name}",
            "estimated_hours": 0.1,
            "data": {},
            "train": {"finetuning_type": "lora"},
            "eval_plan": "sampled eval",
            "expected_signal": "completed score",
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


def _fake_fan_out(*, cfg, study_id, specs, deadline=None):
    """Stand in for the sub-agent: write a completed report.json per spec at the path the loop
    reads, with a stable id (so resume reuses it) and a model path (so the tree chains)."""
    from dataos.core.study import _slug

    exp_root = Path(cfg.workspace) / cfg.project_id / study_id / "experiments"
    reports: dict[str, dict] = {}
    for spec in specs:
        name = spec["name"]
        exp_dir = exp_root / _slug(name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "id": f"exp-{name}",
            "name": name,
            "status": "completed",
            "score": 0.5,
            "model": {"path": str(exp_dir / "final_model")},
        }
        (exp_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        reports[name] = report
    return reports


@pytest.fixture(autouse=True)
def _stub_subagent(monkeypatch):
    monkeypatch.setattr(subagent_executor, "fan_out", _fake_fan_out)


def _cfg(ws):
    return AutoRunConfig(
        project_id="p",
        workspace=ws,
        backend="stub",
        stub=True,
        poll_interval_s=0.05,
        stub_replies={
            "plan": [_plan("n0000", "child-0"), _plan("n0001", "child-1")],
            "review": REVIEW,
        },
        budget=Budget(max_wall_clock_s=1.0),
    )


def _clock_for_rounds(ws, n):
    """Fake clock that trips the wall-clock stop once `n` NEW rounds (review.json written)
    complete in `ws`. Counting completed rounds — not call sites — keeps the test
    independent of how many times the loop calls _clock() per round (budget check +
    live-monitor publishes). The first call is `started` and always reads as in-budget."""
    pdir = Path(ws) / "p"

    def _done() -> int:
        return len(list(pdir.glob("round-*/review.json"))) if pdir.exists() else 0

    state: dict = {"baseline": None}

    def clock() -> float:
        if state["baseline"] is None:
            state["baseline"] = _done()  # resume-safe: pre-existing reviews don't count
            return 0.0
        return 1e9 if _done() - state["baseline"] >= n else 0.0

    return clock


def test_stop_wall_clock_after_n_rounds(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(ws, 2))
    assert res["stop_reason"] == "budget:wall_clock"
    assert res["rounds_completed"] == 2

    pdir = Path(ws) / "p"
    for r, exp in {"round-0000": "child-0", "round-0001": "child-1"}.items():
        assert (pdir / r / "plan.json").is_file()
        assert (pdir / r / "review.json").is_file()
        rep = json.loads((pdir / r / "experiments" / exp / "report.json").read_text())
        assert rep["status"] == "completed"

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
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(ws, 0))
    assert res["stop_reason"] == "budget:wall_clock"
    assert res["rounds_completed"] == 0


def test_best_over_tree_final_model(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(ws, 2))
    fm = res["final_model"]
    assert fm and fm["source"] == "best_over_tree" and fm["score"] == 0.5
    assert (Path(ws) / "p" / "final_model.json").is_file()


def test_resume_skips_done_round(tmp_path):
    ws = str(tmp_path / "ws")
    run_auto(_cfg(ws), _clock=_clock_for_rounds(ws, 1))
    rp = Path(ws) / "p" / "round-0000" / "experiments" / "child-0" / "report.json"
    before = json.loads(rp.read_text())

    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(ws, 1))
    after = json.loads(rp.read_text())
    assert after["id"] == before["id"]
    assert res["rounds_completed"] == 2
    assert (Path(ws) / "p" / "round-0001" / "review.json").is_file()
