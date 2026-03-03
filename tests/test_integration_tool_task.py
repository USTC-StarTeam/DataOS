from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_create_file_task_end_to_end(tmp_path: Path) -> None:
    assert os.getenv("STARTEAM_BASE_URL"), "STARTEAM_BASE_URL must be set for integration tests"
    assert os.getenv("STAETEAM_API_KEY"), "STAETEAM_API_KEY must be set for integration tests"

    target = tmp_path / "agent_task.txt"
    prompt = (
        "Use tools to create file agent_task.txt with exact content: hello world. "
        "Do not add extra punctuation in file content."
    )

    result = subprocess.run(
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
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert target.exists(), f"expected file not created; stdout={result.stdout} stderr={result.stderr}"
    assert target.read_text(encoding="utf-8") == "hello world"
