"""Score salvage: an experimenter cut off AFTER the eval produced a score must not
lose it — the loop rescues the score from the partial report (incremental score-first
write per the contract) or, failing that, the remote summary.json."""

import json

from dataos.agent import report as report_schema
from dataos.agent import runtime
from tests.agent.test_roles import _cfg


class _CutAgent:
    """Dies mid-run (as if the CLI stalled past the deadline) without a terminal report."""

    def __init__(self, on_run=None, **kwargs):
        self._on_run = on_run

    def run(self, task, *, output=None, resume=None, timeout=None):
        if self._on_run:
            self._on_run()
        raise RuntimeError("cli_agent timed out after 10s")


def _run(spec_name="explore-0", on_run=None, cfg=None, tmp_path=None):
    import dataos.agent.roles as m

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
        # the cursor reached `review` — the eval is scored and on disk, the analysis is not written
        # yet — so this reads as mid-run, which is exactly the state salvage exists to finish
        (exp_dir / "report.json").write_text(json.dumps({
            "id": "n0001",
            "status": "review",
            "stages": {
                "eval": {"score": 28.125, "limit": -1},
                "train": {"model_path": "/remote/final_model"},
            },
        }))

    _, report = _run(on_run=write_partial, cfg=_cfg(tmp_path / "ws"), tmp_path=tmp_path)
    parsed = report_schema.read(report)
    assert parsed.status == "completed"
    assert parsed.score == 28.125
    assert report["salvaged"] is True
    assert "partial report.json" in report["error"]
    # a salvaged report must read as FINISHED, or the resume loop re-runs an experiment whose
    # score is already on disk (that is what produced the previous batch's zombie rounds)
    assert report_schema.is_terminal(report)


def test_salvage_from_remote_summary(tmp_path, monkeypatch):
    """No local report at all; the remote eval summary has the score."""
    summary = {"score": 18.08, "task": "gsm8k", "limit": 164}
    eval_dir = "/deploy/dataos_runs/p/round-0000/explore-0/eval/gsm8k_full"

    def _ssh(cfg, cmd, **kw):
        # the executor ASKS which eval dirs exist rather than reconstructing the name: the tag it
        # used to build ("auto" for an AUTO-sized eval) is one no eval has ever written
        if cmd.startswith("ls "):
            return f"{eval_dir}/summary.json"
        return json.dumps(summary) if "summary.json" in cmd else ""

    monkeypatch.setattr(runtime, "_ssh_text", _ssh)
    _, report = _run(cfg=_cfg(tmp_path / "ws"), tmp_path=tmp_path)
    parsed = report_schema.read(report)
    assert parsed.status == "completed"
    assert parsed.score == 18.08
    # the caliber comes from the summary the score came from, never from the requested limit
    assert parsed.stages.eval.limit == 164
    assert report["salvaged"] is True
    assert "remote summary.json" in report["error"]
    assert eval_dir in report["error"]
    assert report_schema.is_terminal(report)


def test_no_score_anywhere_stays_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_ssh_text", lambda cfg, cmd, **kw: "")
    _, report = _run(cfg=_cfg(tmp_path / "ws"), tmp_path=tmp_path)
    parsed = report_schema.read(report)
    assert parsed.status == "failed"
    assert parsed.score is None
    # No eval stage at all. It used to synthesise one at `{"status":"failed","limit":0}`, and that
    # `limit: 0` — a value no real eval can produce, since -1 is full and a sample is positive —
    # was copied out of the matching prompt block by the four nodes lost on 2026-08-15, where it
    # became the fingerprint that identified them. A stage that never ran claims nothing.
    assert parsed.stages.eval is None
    assert report["error"]
    assert "salvaged" not in report
    assert report_schema.is_terminal(report)
