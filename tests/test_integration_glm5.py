from __future__ import annotations

import json
import os
import subprocess
import sys
from urllib import request

import pytest


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    assert value, f"{name} must be set for integration tests"
    return value


@pytest.mark.integration
def test_models_connectivity_with_env_and_glm5_available() -> None:
    base_url = _get_required_env("STARTEAM_BASE_URL").rstrip("/")
    api_key = _get_required_env("STAETEAM_API_KEY")

    req = request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with request.urlopen(req, timeout=30) as resp:
        assert getattr(resp, "status", 200) == 200
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = [item.get("id") for item in data if isinstance(item, dict)]
    assert "glm-5" in model_ids, "glm-5 must be available in /models response"


@pytest.mark.integration
def test_glm5_print_mode_with_env() -> None:
    _get_required_env("STARTEAM_BASE_URL")
    _get_required_env("STAETEAM_API_KEY")

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
