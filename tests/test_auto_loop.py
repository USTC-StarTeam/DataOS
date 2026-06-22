"""Full autonomous skeleton, all-stub (no GPU / no LLM / no claude CLI).

Stop policy is a single wall-clock condition; tests inject a fake monotonic clock so the
loop runs an exact number of rounds before the budget trips. The planner uses the
deterministic `fake` backend whose fixed-JSON reply is parsed back into PlanOut (3 edges,
one per slot). Each round = 3 experiments via the tree's 2-exploit + 1-explore selection.
"""

import json
from pathlib import Path

from dataos_auto.config import AutoRunConfig, Budget
from dataos_auto.loop import run_auto

# one edge per slot (3 slots/round); stub train so run_one needs no GPU/HF
PLAN = json.dumps(
    {
        "goal": "stub round",
        "edges": [
            {"data": {}, "train": {"mode": "stub"}},
            {"data": {}, "train": {"mode": "stub"}},
            {"data": {}, "train": {"mode": "stub"}},
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


# round 0 has no scored nodes -> all explore; round 1 picks top-2 (exploit) + root (explore)
_ROUND_EXPS = {
    "round-0000": ("explore-0", "explore-1", "explore-2"),
    "round-0001": ("exploit-0", "exploit-1", "explore-2"),
}


def test_stop_wall_clock_after_n_rounds(tmp_path):
    ws = str(tmp_path / "ws")
    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(2))
    assert res["stop_reason"] == "budget:wall_clock"
    assert res["rounds_completed"] == 2
    pdir = Path(ws) / "p"
    for r, exps in _ROUND_EXPS.items():
        assert (pdir / r / "plan.json").is_file()
        assert (pdir / r / "review.json").is_file()
        for e in exps:
            rep = json.loads((pdir / r / "experiments" / e / "report.json").read_text())
            assert rep["status"] == "completed"
    # prepared data + agent workspaces live OUTSIDE the project dir (not mis-counted as studies)
    assert (Path(ws) / "_prepared" / "p").exists()
    assert res["summary"]["n_studies"] == 2
    # the externalized tree: root + 3 nodes/round, all scored by the stub eval
    tree = json.loads((pdir / "tree.json").read_text())
    assert len(tree["nodes"]) == 1 + 2 * 3
    scored = [n for n in tree["nodes"] if n["score"] is not None]
    assert len(scored) == 6 and all(n["score"] == 0.5 for n in scored)


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
    run_auto(_cfg(ws), _clock=_clock_for_rounds(1))  # round-0000 only
    rp = Path(ws) / "p" / "round-0000" / "experiments" / "explore-0" / "report.json"
    before = json.loads(rp.read_text())

    res = run_auto(_cfg(ws), _clock=_clock_for_rounds(1))  # resume: runs round-0001
    after = json.loads(rp.read_text())
    assert after["id"] == before["id"]  # round-0000 NOT re-run on resume
    assert res["rounds_completed"] == 2
    assert (Path(ws) / "p" / "round-0001" / "review.json").is_file()
