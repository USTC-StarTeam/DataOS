from __future__ import annotations

import pytest

from dataos.llm.claude_agent_sdk import ClaudeAgentSDKClient
from dataos.runtime.runner import _build_client_from_env


class StubClaudeClient(ClaudeAgentSDKClient):
    async def _run_prompt(self, prompt: str) -> str:
        assert "Return exactly one JSON object" in prompt
        return 'preface {"steps": [{"operator_name": "clean_data", "params": {}}]} trailer'


def test_claude_agent_sdk_client_extracts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAOS_CLAUDE_CLI_PATH", "/tmp/claude")
    client = StubClaudeClient(api_key="k", base_url="https://example.com")

    payload = client.generate_json("system", "user")

    assert payload == {"steps": [{"operator_name": "clean_data", "params": {}}]}


def test_build_client_from_env_prefers_claude_agent_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STARTEAM_API_KEY", "k")
    monkeypatch.setenv("STARTEAM_BASE_URL", "https://example.com")
    monkeypatch.setenv("STARTEAM_MODEL", "demo-model")
    monkeypatch.setenv("DATAOS_CLAUDE_CLI_PATH", "/tmp/claude")

    client = _build_client_from_env()

    assert isinstance(client, ClaudeAgentSDKClient)
    assert client.model == "demo-model"


def test_build_client_from_env_requires_starteam_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STARTEAM_API_KEY", raising=False)
    monkeypatch.delenv("STARTEAM_BASE_URL", raising=False)
    monkeypatch.delenv("STARTEAM_MODEL", raising=False)
    monkeypatch.setenv("DATAOS_CLAUDE_CLI_PATH", "/tmp/claude")

    with pytest.raises(ValueError, match="STARTEAM_API_KEY and STARTEAM_BASE_URL"):
        _build_client_from_env()
