"""Deterministic in-process provider for tests — no subprocess, no network.

Controlled via ``config.extra``:
  emit:   str   -> the final assistant text (default "hello from fake")
  status: str   -> force a terminal status (e.g. "disconnected")
  tools:  bool  -> config.tools; when truthy, also emit one tool_use/tool_result
"""

from __future__ import annotations

from typing import AsyncIterator

from ..types import AgentConfig, Event, Result, Step, Usage
from .base import ProviderSpec, register


class FakeSpec(ProviderSpec):
    name = "fake"

    async def stream(self, config: AgentConfig, prompt: str) -> AsyncIterator[Event]:
        text = config.extra.get("emit", "hello from fake")
        forced_status = config.extra.get("status")
        result = Result(session_id="fake-session")

        if forced_status and forced_status != "done":
            result.status = forced_status
            result.error = f"forced {forced_status}"
            yield Event(type="done", data=result)
            return

        if config.tools:
            yield Event(type="tool_use", name="Bash", args={"command": "echo hi"})
            yield Event(type="tool_result", name="Bash", data="hi")
            result.steps.append(Step(tool="Bash", args={"command": "echo hi"}, result="hi"))

        yield Event(type="text", text=text)
        result.text = text
        result.usage = Usage(input_tokens=5, output_tokens=3, total_tokens=8)
        yield Event(type="done", data=result)


register(FakeSpec())
