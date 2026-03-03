from __future__ import annotations

import json
from pathlib import Path

from py_agent.cli import run


def test_debug_log_uses_default_path(test_workspace: Path, monkeypatch) -> None:
    monkeypatch.chdir(test_workspace)
    rc = run(["--debug", "high", "--version"])
    assert rc == 0
    log_path = test_workspace / ".py-agent" / "logs" / "debug.log"
    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
    messages = [line.get("message", "") for line in lines]
    assert any("debug logging enabled" in msg for msg in messages)
    assert any("exited with --version" in msg for msg in messages)
    assert all("ts" in line and "level" in line for line in lines)


def test_debug_log_uses_custom_path(test_workspace: Path, monkeypatch) -> None:
    monkeypatch.chdir(test_workspace)
    rc = run(["--debug", "high", "--log-path", "./logs/custom.log", "--version"])
    assert rc == 0
    log_path = test_workspace / "logs" / "custom.log"
    assert log_path.exists()


def test_log_path_requires_debug(test_workspace: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(test_workspace)
    rc = run(["--log-path", "./logs/custom.log", "--version"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "requires --debug" in captured.err


def test_debug_low_creates_log_file_without_info_lines(test_workspace: Path, monkeypatch) -> None:
    monkeypatch.chdir(test_workspace)
    rc = run(["--debug", "--version"])
    assert rc == 0
    log_path = test_workspace / ".py-agent" / "logs" / "debug.log"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8").strip()
    # low level and --version produce no runtime events, so file stays empty.
    assert content == ""
