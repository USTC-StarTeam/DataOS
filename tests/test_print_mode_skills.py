from __future__ import annotations

from pathlib import Path
from typing import Any

from py_agent.modes.print_mode import run_print_mode
from py_agent.providers.base import ChatResult, Provider


class CapturingProvider(Provider):
    def __init__(self) -> None:
        self.last_messages: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], model: str, tools=None) -> ChatResult:
        self.last_messages = messages
        return ChatResult(model=model, text="ok", tool_calls=[])


def test_print_mode_injects_skills_system_prompt_and_expands_invocation(test_workspace: Path) -> None:
    skills_root = test_workspace / "skills" / "writer"
    skills_root.mkdir(parents=True, exist_ok=True)
    (skills_root / "SKILL.md").write_text("# writer\nWrite concise markdown.", encoding="utf-8")

    provider = CapturingProvider()
    run_print_mode(
        provider=provider,
        prompt="/skill:writer create changelog",
        model="glm-5",
        mode="text",
        cwd=str(test_workspace),
    )

    assert len(provider.last_messages) >= 2
    assert provider.last_messages[0]["role"] == "system"
    assert "Available skills" in str(provider.last_messages[0]["content"])
    assert "writer" in str(provider.last_messages[0]["content"])
    assert provider.last_messages[1]["role"] == "user"
    assert "/skill:writer" not in str(provider.last_messages[1]["content"])
    assert "Write concise markdown." in str(provider.last_messages[1]["content"])
