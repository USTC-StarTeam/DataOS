from __future__ import annotations

import pytest

from py_agent.providers.base import ProviderError
from py_agent.providers.openai_responses import OpenAICompatibleProvider


def test_provider_requires_env_or_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STARTEAM_BASE_URL", raising=False)
    monkeypatch.delenv("STAETEAM_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenAICompatibleProvider(base_url="", api_key="")


def test_provider_parses_chat_completion_response() -> None:
    def fake_transport(url: str, payload: dict, headers: dict) -> dict:
        assert url == "https://example.com/v1/chat/completions"
        assert payload["model"] == "glm-5"
        assert payload["messages"][0]["content"] == "hi"
        assert headers["Authorization"] == "Bearer token"
        return {"choices": [{"message": {"content": "hello"}}]}

    provider = OpenAICompatibleProvider(base_url="https://example.com", api_key="token", transport=fake_transport)
    result = provider.complete(prompt="hi", model="glm-5")
    assert result.text == "hello"


def test_provider_uses_single_v1_path_when_base_url_has_v1() -> None:
    seen = {}

    def fake_transport(url: str, payload: dict, headers: dict) -> dict:
        seen["url"] = url
        return {"choices": [{"message": {"content": "ok"}}]}

    provider = OpenAICompatibleProvider(
        base_url="https://example.com/v1",
        api_key="token",
        transport=fake_transport,
    )
    provider.complete(prompt="hi", model="glm-5")
    assert seen["url"] == "https://example.com/v1/chat/completions"


def test_provider_high_debug_logs_request_and_response_with_redacted_auth() -> None:
    logs: list[str] = []

    class StubLogger:
        def llm(self, message: str) -> None:
            logs.append(message)

    def fake_transport(url: str, payload: dict, headers: dict) -> dict:
        return {"choices": [{"message": {"content": "ok"}}]}

    provider = OpenAICompatibleProvider(
        base_url="https://example.com",
        api_key="secret-token",
        transport=fake_transport,
        debug_logger=StubLogger(),  # type: ignore[arg-type]
        debug_level="high",
    )
    provider.complete(prompt="hi", model="glm-5")
    joined = "\n".join(logs)
    assert "request url=https://example.com/v1/chat/completions" in joined
    assert "response body=" in joined
    assert "Bearer ***" in joined
    assert "secret-token" not in joined


def test_provider_retries_with_compat_messages_on_tool_role_pairing_error() -> None:
    seen_payloads: list[dict] = []

    def fake_transport(url: str, payload: dict, headers: dict) -> dict:
        seen_payloads.append(payload)
        if len(seen_payloads) == 1:
            raise ProviderError(
                'request failed: HTTP 400 body={"error":{"message":"messages with role \\"tool\\" must be a response"}}'
            )
        return {"choices": [{"message": {"content": "ok"}}]}

    provider = OpenAICompatibleProvider(base_url="https://example.com", api_key="token", transport=fake_transport)
    result = provider.chat(
        model="glm-5",
        tools=[],
        messages=[
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"pwd\"}"}}
                ],
            },
            {"role": "tool", "name": "bash", "tool_call_id": "t1", "content": "ok"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
        ],
    )
    assert result.text == "ok"
    assert len(seen_payloads) == 2
    retry_messages = seen_payloads[1]["messages"]
    assert all(m.get("role") != "tool" for m in retry_messages)
    assert not any(m.get("tool_calls") is not None for m in retry_messages if m.get("role") == "assistant")
