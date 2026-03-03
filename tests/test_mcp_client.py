from __future__ import annotations

import sys
from pathlib import Path

import pytest

from py_agent.mcp.client import McpClientError, StdioMcpClient


def test_stdio_mcp_client_list_and_call() -> None:
    server_script = Path("tests/fixtures/mcp/mock_server.py").resolve()
    client = StdioMcpClient(
        server_name="demo",
        command=sys.executable,
        args=[str(server_script)],
        env={},
        cwd=str(server_script.parent.parent.parent),
    )

    client.start()
    try:
        tools = client.list_tools()
        names = [t["name"] for t in tools]
        assert "echo" in names
        assert "add" in names

        assert client.call_tool("echo", {"text": "hello"}) == "hello"
        assert client.call_tool("add", {"a": 2, "b": 3}) == "5"
    finally:
        client.close()


def test_stdio_mcp_client_unknown_tool_raises() -> None:
    server_script = Path("tests/fixtures/mcp/mock_server.py").resolve()
    client = StdioMcpClient(
        server_name="demo",
        command=sys.executable,
        args=[str(server_script)],
        env={},
        cwd=str(server_script.parent.parent.parent),
    )
    client.start()
    try:
        with pytest.raises(McpClientError):
            client.call_tool("missing_tool", {})
    finally:
        client.close()
