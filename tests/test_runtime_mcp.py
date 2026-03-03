from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from py_agent.mcp.registry import McpRegistry
from py_agent.providers.base import ChatResult, Provider, ToolCall
from py_agent.runtime import run_agent_loop
from py_agent.tools import create_local_tools


@dataclass
class McpProvider(Provider):
    calls: int = 0

    def chat(self, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]] | None = None) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(
                model=model,
                text="",
                tool_calls=[ToolCall(id="m1", name="mcp.demo.echo", arguments={"text": "from-mcp"})],
            )
        return ChatResult(model=model, text="done", tool_calls=[])


def test_runtime_can_execute_mcp_tool(test_workspace: Path) -> None:
    server_script = Path("tests/fixtures/mcp/mock_server.py").resolve()
    cfg = test_workspace / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "servers": {
                    "demo": {
                        "command": sys.executable,
                        "args": [str(server_script)],
                        "cwd": str(server_script.parent.parent.parent),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    registry = McpRegistry(config_path=cfg, enabled_servers=["demo"])
    registry.start()
    try:
        tools = create_local_tools(str(test_workspace))
        tools.update(registry.build_tools())
        result = run_agent_loop(provider=McpProvider(), model="glm-5", prompt="run mcp tool", tools=tools)
        assert result.text == "done"
        tool_results = [e for e in result.events if e.get("type") == "tool_result"]
        assert tool_results
        assert "from-mcp" in tool_results[0]["content"]
    finally:
        registry.close()
