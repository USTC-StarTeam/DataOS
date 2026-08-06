"""Full agentic loop, all-stub (no GPU / no LLM / no claude CLI).

The loop's only executor is the experimenter (``dataos.agent.subagent_executor``); here we
replace its ``fan_out`` with a fake that just writes a completed ``report.json`` per spec, so
the test exercises the loop's orchestration (plan → execute → review → tree → budget stop →
resume) without a real experimenter / H100.
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
    """Stand in for the experimenter: write a completed report.json per spec at the path the loop
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
        # round 0 is the code-fixed baseline (no leader call); the leader's first plan
        # comes at round 1 and should parent on the baseline node n0001.
        stub_replies={
            "plan": [_plan("n0001", "child-0"), _plan("n0002", "child-1")],
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
    for r, exp in {"round-0000": "baseline", "round-0001": "child-0"}.items():
        assert (pdir / r / "plan.json").is_file()
        assert (pdir / r / "review.json").is_file()
        rep = json.loads((pdir / r / "experiments" / exp / "report.json").read_text())
        assert rep["status"] == "completed"

    assert res["summary"]["n_studies"] == 2

    tree = json.loads((pdir / "tree.json").read_text())
    assert len(tree["nodes"]) == 3
    assert tree["nodes"][1]["parent"] == "n0000"  # baseline on the root
    assert tree["nodes"][2]["parent"] == "n0001"  # child-0 continues from the baseline
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
    rp = Path(ws) / "p" / "round-0000" / "experiments" / "baseline" / "report.json"
    before = json.loads(rp.read_text())

    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(ws, 1))
    after = json.loads(rp.read_text())
    assert after["id"] == before["id"]
    assert res["rounds_completed"] == 2
    assert (Path(ws) / "p" / "round-0001" / "review.json").is_file()


def test_validate_plan_source_gate():
    """The source gate checks EXISTENCE in recon profiles only (no quality labels)."""
    from dataos.agent.loop import _validate_plan
    from dataos.agent.schemas import ExperimentPlan
    import types

    recon = {"source_profiles": [
        {"source": "good_math", "domain": "math", "n": 1000,
         "target_fit": {"target_ratio": 0.0, "negatives": 0.9}},  # ugly stats, but allowed
    ]}
    plan = ExperimentPlan(
        parent_node_id="n0000", name="x", hypothesis="h", estimated_hours=1.0,
        data={"sources": ["good_math"], "recipe": []}, train={"finetuning_type": "lora"},
        eval_plan="e", expected_signal="s", subagent_brief="b", rationale="r",
    )
    tree = types.SimpleNamespace(
        has_node=lambda _id: True,
        growable_nodes=lambda: [{"id": "n0000"}],
        experiment_names=lambda: set(),
    )
    cfg = types.SimpleNamespace(target_bench="gsm8k")
    # a profiled source passes even with ratio 0.0 / 90% negatives (leader's call, not code's)
    assert _validate_plan(plan, tree, cfg, recon) == []
    # an unprofiled (invented) source is rejected
    plan.data["sources"] = ["invented_source"]
    errors = _validate_plan(plan, tree, cfg, recon)
    assert any("not in recon.source_profiles" in e for e in errors)


def test_no_function_local_subagent_executor_import():
    """Regression: a function-local `from dataos.agent import subagent_executor` inside
    run_auto shadowed the module-level import and crashed --resume (recon present -> the
    recon branch's local import never ran -> the budget line hit an unbound local)."""
    import inspect

    import dataos.agent.loop as loop

    for name, fn in inspect.getmembers(loop, inspect.isfunction):
        src = inspect.getsource(fn)
        assert "from dataos.agent import subagent_executor" not in src, (
            f"{name} has a function-local subagent_executor import — it shadows the "
            "module-level import and breaks the resume path"
        )
