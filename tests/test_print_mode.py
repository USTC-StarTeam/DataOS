from __future__ import annotations

import json

from py_agent.modes.print_mode import run_print_mode
from py_agent.providers.base import CompletionResult, Provider


class DummyProvider(Provider):
    def complete(self, prompt: str, model: str) -> CompletionResult:
        assert prompt == "hello"
        assert model == "glm-5"
        return CompletionResult(model=model, text="world")


def test_print_mode_text_output() -> None:
    result = run_print_mode(provider=DummyProvider(), prompt="hello", model="glm-5", mode="text")
    assert result.output == "world"


def test_print_mode_json_output() -> None:
    result = run_print_mode(provider=DummyProvider(), prompt="hello", model="glm-5", mode="json")
    payload = json.loads(result.output)
    assert payload["type"] == "final"
    assert payload["model"] == "glm-5"
    assert payload["content"] == "world"
