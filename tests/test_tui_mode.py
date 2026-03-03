from __future__ import annotations

from typing import Any

from py_agent.modes.tui_mode import format_event_lines, run_tui_mode
from py_agent.providers.base import ChatResult, Provider


class EchoProvider(Provider):
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], model: str, tools=None) -> ChatResult:
        self.calls += 1
        user_messages = [m for m in messages if m.get("role") == "user"]
        text = str(user_messages[-1].get("content", "")) if user_messages else ""
        return ChatResult(model=model, text=f"echo:{text}", tool_calls=[])


def test_tui_mode_single_roundtrip_then_exit(test_workspace) -> None:
    provider = EchoProvider()
    outputs: list[str] = []
    user_inputs = iter(["hello", "/exit"])

    def fake_input(_: str) -> str:
        return next(user_inputs)

    rc = run_tui_mode(
        provider=provider,
        model="glm-5",
        cwd=str(test_workspace),
        input_func=fake_input,
        output_func=outputs.append,
    )
    assert rc == 0
    assert provider.calls == 1
    assert any("echo:hello" in line for line in outputs)


def test_format_event_lines_renders_tool_events() -> None:
    events = [
        {"type": "tool_call", "tool": "read", "tool_call_id": "t1", "arguments": {"path": "a.txt"}},
        {"type": "tool_result", "tool": "read", "tool_call_id": "t1", "content": "hello"},
    ]
    lines = format_event_lines(events)
    assert lines[0] == "[tool_call] read id=t1 args={'path': 'a.txt'}"
    assert lines[1] == "[tool_result] read id=t1 content=hello"

