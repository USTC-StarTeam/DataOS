"""The benchmark's real size, and the smallest subset a probe may use.

``limit`` in an eval summary is only the REQUEST (-1 means "all"), so until the round-0 baseline
records ``eval.n_samples`` the run has no idea what a subset score is a subset of. Everything
that judges a subset — the probe floor, the plan gate, the "@N/full" annotation — hangs off it.
"""

import json
from pathlib import Path

import pytest

from dataos.agent.config import AutoRunConfig
from dataos.agent.loop import _full_eval_size, _probe_floor


def _baseline(tmp_path: Path, eval_block: dict | None) -> AutoRunConfig:
    cfg = AutoRunConfig(project_id="p", workspace=str(tmp_path), target_bench="gsm8k")
    d = tmp_path / "p" / "round-0000" / "experiments" / "baseline"
    d.mkdir(parents=True, exist_ok=True)
    report = {"id": "n0001", "status": "completed", "stages": {"eval": dict(eval_block or {})}}
    (d / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return cfg


def test_full_size_comes_from_the_baseline(tmp_path):
    cfg = _baseline(tmp_path, {"score": 20.0, "limit": -1, "n_samples": 1319})
    assert _full_eval_size(cfg) == 1319


def test_full_size_is_none_before_the_baseline_lands(tmp_path):
    cfg = AutoRunConfig(project_id="p", workspace=str(tmp_path), target_bench="gsm8k")
    assert _full_eval_size(cfg) is None


def test_full_size_is_none_for_a_pre_field_baseline(tmp_path):
    """Older runs recorded eval.limit but not the item count — say unknown, don't guess."""
    assert _full_eval_size(_baseline(tmp_path, {"score": 20.0, "limit": -1})) is None
    assert _full_eval_size(_baseline(tmp_path, None)) is None


@pytest.mark.parametrize(
    "full,floor",
    [
        (1319, 439),   # a third of a large benchmark: a probe saves two thirds of the cost
        (500, 166),
        (300, 100),
        (164, 54),     # was lifted to 100 (61% of the set) by the old constant floor
        (100, 33),     # bfcl: the old floor equalled the full size, disabling subsetting
        (30, 10),      # aime2025: same, and "at least 100 items" is incoherent for 30
        (60, 20),
    ],
)
def test_probe_floor(full, floor):
    assert _probe_floor(full) == floor


def test_probe_floor_unknown_size_gives_no_guidance(tmp_path):
    """No numeric floor is better than a made-up one — the gate falls back to its old check."""
    assert _probe_floor(None) is None
    assert _probe_floor(0) is None


def _plan(eval_limit):
    from types import SimpleNamespace
    return SimpleNamespace(parent_node_id="n0000", name="e1", train={}, data={},
                           estimated_hours=1.0, eval_limit=eval_limit)


def _auto_cfg(tmp_path, n_samples):
    cfg = _baseline(tmp_path, {"score": 20.0, "limit": -1, "n_samples": n_samples})
    cfg.eval_limit = -2  # AUTO: the leader picks each experiment's eval size
    return cfg


def _errors(tmp_path, *, n_samples, eval_limit):
    from dataos.agent.loop import _validate_plan
    from dataos.agent.state_tree import TreeState
    tree = TreeState.load(tmp_path / "tree", "base")
    return [e for e in _validate_plan(_plan(eval_limit), tree, _auto_cfg(tmp_path, n_samples))
            if "eval_limit" in e]


def test_probe_below_the_floor_is_rejected(tmp_path):
    """The 4b2 batch lost benches to 5- and 8-item probes; the floor makes that unplannable."""
    errs = _errors(tmp_path, n_samples=1319, eval_limit=8)
    assert errs and "probe floor 439" in errs[0]


def test_probe_at_the_floor_and_full_eval_are_accepted(tmp_path):
    assert not _errors(tmp_path, n_samples=1319, eval_limit=439)
    assert not _errors(tmp_path, n_samples=1319, eval_limit=-1)


def test_no_floor_known_falls_back_to_the_old_check(tmp_path):
    """A run whose baseline never recorded the size keeps its previous freedom, not a guess."""
    assert not _errors(tmp_path, n_samples=None, eval_limit=8)
