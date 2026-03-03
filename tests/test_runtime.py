from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from py_agent.providers.base import ChatResult, Provider, ProviderError, ToolCall
from py_agent.runtime import run_agent_loop
from py_agent.tools import create_local_tools


@dataclass
class FakeProvider(Provider):
    calls: int = 0

    def chat(self, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]] | None = None) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(
                model=model,
                text="",
                tool_calls=[ToolCall(id="1", name="write", arguments={"path": "x.txt", "content": "OK"})],
            )
        return ChatResult(model=model, text="done", tool_calls=[])


def test_runtime_executes_tool_then_returns_final_text(test_workspace: Path) -> None:
    provider = FakeProvider()
    tools = create_local_tools(str(test_workspace))
    result = run_agent_loop(provider=provider, model="glm-5", prompt="write file", tools=tools)
    assert result.text == "done"
    assert (test_workspace / "x.txt").read_text() == "OK"


@dataclass
class FallbackProvider(Provider):
    calls: int = 0

    def chat(self, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]] | None = None) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(
                model=model,
                text="",
                tool_calls=[ToolCall(id="1", name="write", arguments={"path": "y.txt", "content": "Y"})],
            )
        raise ProviderError('request failed: HTTP 400 body={"error":{"message":"messages with role \\"tool\\""}}')


def test_runtime_falls_back_to_tool_output_on_gateway_tool_message_rejection(test_workspace: Path) -> None:
    provider = FallbackProvider()
    tools = create_local_tools(str(test_workspace))
    result = run_agent_loop(provider=provider, model="glm-5", prompt="write file", tools=tools)
    assert "wrote" in result.text
    assert (test_workspace / "y.txt").read_text() == "Y"
