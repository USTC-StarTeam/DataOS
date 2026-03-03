from __future__ import annotations

import subprocess
import sys


def run_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "py_agent.main", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_version_flag_outputs_version() -> None:
    result = run_cmd("--version")
    assert result.returncode == 0
    assert result.stdout.strip() == "0.1.0"


def test_help_flag_outputs_usage() -> None:
    result = run_cmd("--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "--version" in result.stdout


def test_print_mode_without_prompt_fails() -> None:
    result = run_cmd("-p")
    assert result.returncode == 2
    assert "missing prompt" in result.stderr.lower()
