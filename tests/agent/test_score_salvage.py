"""Score salvage: an experimenter cut off AFTER the eval produced a score must not
lose it — the loop rescues the score from the partial report (incremental score-first
write per the contract) or, failing that, the remote summary.json."""

import json

from dataos.agent import subagent_executor as subagent_exec
from tests.agent.test_subagent_executor import _cfg


class _CutAgent:
    """Dies mid-run (as if the CLI stalled past the deadline) without a terminal report."""

    def __init__(self, on_run=None, **kwargs):
        self._on_run = on_run

    def run(self, task, *, output=None, resume=None, timeout=None):
        if self._on_run:
            self._on_run()
        raise RuntimeError("cli_agent timed out after 10s")


def _run(spec_name="explore-0", on_run=None, cfg=None, tmp_path=None):
    import dataos.agent.subagent_executor as m

    orig_build = m.build_agent
    m.build_agent = lambda *a, **k: _CutAgent(on_run=on_run, **k)
    orig_sleep = m.time.sleep
    m.time.sleep = lambda _s: None
    try:
        spec = {"name": spec_name, "data": {}, "train": {}, "init_model": "/models/parent"}
        return m._run_one(cfg, "round-0000", spec, tmp_path / "experiments", timeout=10)
    finally:
        m.build_agent = orig_build
        m.time.sleep = orig_sleep


def test_salvage_from_partial_report(tmp_path):
    """Score-first incremental write + cut mid-analysis -> completed + salvaged."""
    exp_dir = tmp_path / "experiments" / "explore-0"

    def write_partial():
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "report.json").write_text(json.dumps({
            "name": "explore-0", "status": "running", "score": 28.125,
            "stages": {"eval": {"status": "completed"}},
            "model": {"path": "/remote/final_model"},
        }))

    _, report = _run(on_run=write_partial, cfg=_cfg(tmp_path / "ws"), tmp_path=tmp_path)
    assert report["status"] == "completed"
    assert report["score"] == 28.125
    assert report["salvaged"] is True
    assert "partial report.json" in report["error"]


def test_salvage_from_remote_summary(tmp_path, monkeypatch):
    """No local report at all; the remote eval summary has the score."""
    summary = {"score": 18.08, "task": "gsm8k"}
    monkeypatch.setattr(
        subagent_exec, "_ssh_text",
        lambda cfg, cmd, **kw: json.dumps(summary) if "summary.json" in cmd else "",
    )
    _, report = _run(cfg=_cfg(tmp_path / "ws"), tmp_path=tmp_path)
    assert report["status"] == "completed"
    assert report["score"] == 18.08
    assert report["salvaged"] is True
    assert "remote summary.json" in report["error"]
    assert report["eval"]["path"].endswith("eval/gsm8k_limit20")


def test_no_score_anywhere_stays_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(subagent_exec, "_ssh_text", lambda cfg, cmd, **kw: "")
    _, report = _run(cfg=_cfg(tmp_path / "ws"), tmp_path=tmp_path)
    assert report["status"] == "failed"
    assert report["score"] is None
    assert "salvaged" not in report
