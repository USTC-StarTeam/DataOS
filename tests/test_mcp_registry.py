from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from py_agent.mcp.registry import McpRegistry, McpRegistryError


def test_mcp_registry_builds_namespaced_tools(test_workspace: Path) -> None:
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
        tools = registry.build_tools()
        assert "mcp.demo.echo" in tools
        assert "mcp.demo.add" in tools
        assert tools["mcp.demo.echo"].execute({"text": "x"}) == "x"
        assert tools["mcp.demo.add"].execute({"a": 3, "b": 4}) == "7"
    finally:
        registry.close()


def test_mcp_registry_unknown_enabled_server_raises(test_workspace: Path) -> None:
    cfg = test_workspace / "mcp.json"
    cfg.write_text(json.dumps({"servers": {}}), encoding="utf-8")
    registry = McpRegistry(config_path=cfg, enabled_servers=["missing"])
    with pytest.raises(McpRegistryError):
        registry.start()
