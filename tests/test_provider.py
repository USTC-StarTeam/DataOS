from __future__ import annotations

import pytest

from py_agent.providers.base import ProviderError
from py_agent.providers.openai_responses import OpenAICompatibleProvider


def test_provider_requires_env_or_args() -> None:
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
