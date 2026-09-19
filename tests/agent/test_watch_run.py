"""The host watchdog must preserve run scope and the original deadline."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/watch_run.py"


@pytest.fixture
def watchdog():
    spec = importlib.util.spec_from_file_location("watch_run_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job_scope_does_not_include_siblings(watchdog, tmp_path):
    root = tmp_path / "run"
    assert watchdog.owned_job(str(root / "round-0001" / "experiment"), root)
    assert not watchdog.owned_job(str(tmp_path / "run-other"), root)
    assert not watchdog.owned_job(str(root / ".." / "other"), root)
    assert not watchdog.owned_job("", root)


def test_restart_spends_only_remaining_budget(watchdog, tmp_path, monkeypatch):
    (tmp_path / "intent.json").write_text('{"deadline_epoch": 7300}')
    monkeypatch.setattr(watchdog.time, "time", lambda: 100)
    monkeypatch.setattr(watchdog.os, "chdir", lambda _: None)
    calls = []
    monkeypatch.setattr(watchdog.os, "execv", lambda exe, argv: calls.append(argv))
    watchdog.run_controller({"budget": {"time_hours": 24}}, tmp_path)
    active = json.loads((tmp_path / "active_config.json").read_text())
    assert active["budget"]["time_hours"] == 2
    assert calls[0][-1] == "--resume"


def test_expired_budget_never_restarts(watchdog, tmp_path, monkeypatch):
    (tmp_path / "intent.json").write_text('{"deadline_epoch": 100}')
    monkeypatch.setattr(watchdog.time, "time", lambda: 101)
    monkeypatch.setattr(watchdog.os, "execv", lambda *a: pytest.fail("started after deadline"))
    watchdog.run_controller({"budget": {"time_hours": 24}}, tmp_path)
    assert not (tmp_path / "active_config.json").exists()


def _progress_fixture(tmp_path):
    workspace = tmp_path / "run"
    experiment = workspace / "round-0001/experiments/trial"
    experiment.mkdir(parents=True)
    report = experiment / "report.json"
    report.write_text(json.dumps({"id": "n0002", "status": "data"}))
    state_dir = tmp_path / "watchdog"
    state_dir.mkdir()
    args = SimpleNamespace(state_dir=state_dir, service="test.service")
    live = {"round": 1, "node_id": "n0002", "phase": "executing", "node_trains": True}
    state = {"remote": {"jobs": [], "judges": []}, "actions": [],
             "service": {"ActiveState": "active"}, "status": "running"}
    return workspace, report, args, live, state


@pytest.mark.parametrize("verified", [True, False])
def test_completion_repair_requires_evaluation_evidence(watchdog, tmp_path, monkeypatch, verified):
    workspace, path, args, live, state = _progress_fixture(tmp_path)
    original = {"id": "n0002", "status": "complete", "stages": {
        "data": {"fixes": "Converted roles"},
        "train": {"model_path": "/remote/model"},
        "eval": {"score": 62.72, "n_samples": 245, "summary_dir": "/remote/eval"},
    }}
    path.write_text(json.dumps(original))
    monkeypatch.setattr(watchdog, "completed_evidence", lambda *a: {
        "model_exists": True,
        "summary": {"score": 62.7200948, "n_samples": 245, "needs_judge": not verified},
    })
    watchdog.check_progress({}, {}, args, workspace, live, state)
    current = json.loads(path.read_text())
    if verified:
        assert current["status"] == "completed"
        assert current["stages"]["data"]["fixes"] == ["Converted roles"]
        assert current["stages"]["eval"] == original["stages"]["eval"]
        assert state["actions"] == ["repaired_report_schema"]
        backup = next(path.parent.glob("report.before-watchdog-*.json"))
        assert json.loads(backup.read_text()) == original
    else:
        assert current == original
        assert state["status"] == "invalid_report"


def test_done_loop_recovers_once_then_pauses(watchdog, tmp_path, monkeypatch):
    workspace, path, args, live, state = _progress_fixture(tmp_path)
    calls = []
    clock = [100.0]
    monkeypatch.setattr(watchdog.time, "time", lambda: clock[0])
    monkeypatch.setattr(watchdog, "command", lambda argv, **kw: calls.append(argv))
    watchdog.check_progress({}, {}, args, workspace, live, state)
    for timestamp in (1300.0, 2500.0, 4900.0):
        clock[0] = timestamp
        live["heartbeat"] = timestamp  # Fresh heartbeats must not hide a stuck experiment.
        with (path.parent / "agent.jsonl").open("a") as stream:
            stream.write(json.dumps({"type": "text", "text": "Done.", "ts": timestamp}) + "\n")
        watchdog.check_progress({}, {}, args, workspace, live, state)
    assert [argv[2] for argv in calls] == ["restart", "stop"]
    assert state["status"] == "needs_attention"
    assert (args.state_dir / "attention.json").exists()


@pytest.mark.parametrize("work", ["jobs", "judges"])
def test_active_remote_work_is_not_mistaken_for_a_stall(watchdog, tmp_path, monkeypatch, work):
    workspace, path, args, live, state = _progress_fixture(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(watchdog.time, "time", lambda: clock[0])
    monkeypatch.setattr(watchdog, "command", lambda *a, **kw: pytest.fail("interrupted remote work"))
    watchdog.check_progress({}, {}, args, workspace, live, state)
    clock[0] += 7200
    state["remote"][work] = [{"state": "RUNNING"}]
    watchdog.check_progress({}, {}, args, workspace, live, state)
    assert state["progress"]["idle_seconds"] == 0
    assert state["actions"] == []


def test_real_report_progress_resets_idle_time(watchdog, tmp_path, monkeypatch):
    workspace, path, args, live, state = _progress_fixture(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(watchdog.time, "time", lambda: clock[0])
    monkeypatch.setattr(watchdog, "command", lambda *a, **kw: pytest.fail("interrupted progress"))
    watchdog.check_progress({}, {}, args, workspace, live, state)
    clock[0] += 2500
    path.write_text(json.dumps({"id": "n0002", "status": "train", "stages": {"data": {"rows": 10}}}))
    watchdog.check_progress({}, {}, args, workspace, live, state)
    assert state["progress"]["idle_seconds"] == 0


def test_unknown_schema_error_pauses_without_guessing_a_repair(watchdog, tmp_path, monkeypatch):
    workspace, path, args, live, state = _progress_fixture(tmp_path)
    original = {"status": "completed", "stages": {"eval": {"score": "not-a-score"}}}
    path.write_text(json.dumps(original))
    clock = [100.0]
    calls = []
    monkeypatch.setattr(watchdog.time, "time", lambda: clock[0])
    monkeypatch.setattr(watchdog, "command", lambda argv, **kw: calls.append(argv))
    watchdog.check_progress({}, {}, args, workspace, live, state)
    clock[0] += 2400
    watchdog.check_progress({}, {}, args, workspace, live, state)
    assert json.loads(path.read_text()) == original
    assert [argv[2] for argv in calls] == ["stop"]
    assert state["status"] == "needs_attention"
