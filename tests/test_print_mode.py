from __future__ import annotations

import json
from typing import Any

from py_agent.modes.print_mode import run_print_mode
from py_agent.providers.base import ChatResult, Provider


class DummyProvider(Provider):
    def chat(self, messages: list[dict[str, Any]], model: str, tools=None) -> ChatResult:
        assert messages[0]["content"] == "hello"
        assert model == "glm-5"
        return ChatResult(model=model, text="world", tool_calls=[])


def test_print_mode_text_output() -> None:
    result = run_print_mode(provider=DummyProvider(), prompt="hello", model="glm-5", mode="text")
    assert result.output == "world"


def test_print_mode_json_output() -> None:
    result = run_print_mode(provider=DummyProvider(), prompt="hello", model="glm-5", mode="json")
    payload = json.loads(result.output)
    assert payload["type"] == "final"
    assert payload["model"] == "glm-5"
    assert payload["content"] == "world"
