from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from py_agent.providers.base import ChatResult, Provider, ToolCall
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


def test_runtime_executes_tool_then_returns_final_text(tmp_path) -> None:
    provider = FakeProvider()
    tools = create_local_tools(str(tmp_path))
    result = run_agent_loop(provider=provider, model="glm-5", prompt="write file", tools=tools)
    assert result.text == "done"
    assert (tmp_path / "x.txt").read_text() == "OK"
