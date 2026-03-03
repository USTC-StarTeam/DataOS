from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_glm5_print_mode_with_env() -> None:
    if not os.getenv("STARTEAM_BASE_URL") or not os.getenv("STAETEAM_API_KEY"):
        pytest.skip("integration env vars not set")

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
            "Reply with exactly: OK",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != ""
