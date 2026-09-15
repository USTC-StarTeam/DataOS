"""The host watchdog must preserve run scope and the original deadline."""

import importlib.util
import json
from pathlib import Path

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
