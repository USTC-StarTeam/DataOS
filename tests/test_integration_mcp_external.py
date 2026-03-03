from __future__ import annotations

import pytest

from py_agent.mcp.client import StdioMcpClient

SERVER_NAME = "external"
COMMAND = "npx"
ARGS = ["-y", "@modelcontextprotocol/server-everything"]
TOOL_NAME = "echo"
TOOL_ARGS = {"message": "hello from external mcp"}
EXPECT_SUBSTRING = "hello from external mcp"


@pytest.mark.mcp_external
def test_external_mcp_tool_list_contains_target() -> None:
    client = StdioMcpClient(
        server_name=SERVER_NAME,
        command=COMMAND,
        args=ARGS,
    )
    client.start()
    try:
        tools = client.list_tools()
        names = [t.get("name") for t in tools if isinstance(t, dict)]
        assert TOOL_NAME in names, f"tool '{TOOL_NAME}' not listed by MCP server"
    finally:
        client.close()


@pytest.mark.mcp_external
def test_external_mcp_tool_call() -> None:
    client = StdioMcpClient(
        server_name=SERVER_NAME,
        command=COMMAND,
        args=ARGS,
    )
    client.start()
    try:
        output = client.call_tool(TOOL_NAME, TOOL_ARGS)
        assert output.strip(), "external MCP tool output is empty"
        assert EXPECT_SUBSTRING in output
    finally:
        client.close()
