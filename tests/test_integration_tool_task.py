from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _require_env() -> None:
    assert os.getenv("STARTEAM_BASE_URL"), "STARTEAM_BASE_URL must be set for integration tests"
    assert os.getenv("STAETEAM_API_KEY"), "STAETEAM_API_KEY must be set for integration tests"


def _run_agent_task(cwd: Path, prompt: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "py_agent.main",
            "-p",
            "--mode",
            "text",
            "--model",
            "glm-5",
            prompt,
        ],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.task_eval
def test_create_file_task_end_to_end(tmp_path: Path) -> None:
    _require_env()
    target = tmp_path / "agent_task.txt"
    prompt = (
        "Use tools to create file agent_task.txt with exact content: hello world. "
        "Do not add extra punctuation in file content."
    )
    result = _run_agent_task(tmp_path, prompt)
    assert result.returncode == 0, result.stderr
    assert target.exists(), f"expected file not created; stdout={result.stdout} stderr={result.stderr}"
    assert target.read_text(encoding="utf-8") == "hello world"


@pytest.mark.task_eval
def test_create_markdown_file_task(tmp_path: Path) -> None:
    _require_env()
    target = tmp_path / "report.md"
    prompt = (
        "Use write tool to create report.md with exact content: # Report\\nstatus: ok\\n"
    )
    result = _run_agent_task(tmp_path, prompt)
    assert result.returncode == 0, result.stderr
    assert target.exists(), f"expected markdown file not created; stdout={result.stdout} stderr={result.stderr}"
    assert target.read_text(encoding="utf-8").strip() == "# Report\nstatus: ok"


@pytest.mark.task_eval
def test_bash_append_content_task(tmp_path: Path) -> None:
    _require_env()
    target = tmp_path / "append.txt"
    target.write_text("line1\n", encoding="utf-8")
    prompt = (
        "Use bash tool only. Run command to append line2 to append.txt "
        "so the file becomes two lines: line1 and line2."
    )
    result = _run_agent_task(tmp_path, prompt)
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == "line1\nline2\n"


@pytest.mark.task_eval
def test_read_file_task_returns_exact_content(tmp_path: Path) -> None:
    _require_env()
    src = tmp_path / "source.txt"
    src.write_text("copy-target-123", encoding="utf-8")
    prompt = (
        "You must use read tool on source.txt and output exactly the file content only."
    )
    result = _run_agent_task(tmp_path, prompt)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "copy-target-123"


@pytest.mark.task_eval
def test_bash_tool_creates_file_task(tmp_path: Path) -> None:
    _require_env()
    target = tmp_path / "bash_out.txt"
    prompt = (
        "Use bash tool to run command: printf 'from-bash' > bash_out.txt. "
        "Do not use write/edit tools for this task."
    )
    result = _run_agent_task(tmp_path, prompt)
    assert result.returncode == 0, result.stderr
    assert target.exists(), f"expected bash output file not created; stdout={result.stdout} stderr={result.stderr}"
    assert target.read_text(encoding="utf-8") == "from-bash"
