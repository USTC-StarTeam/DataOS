from __future__ import annotations

import json
import os

import pytest

from py_agent.mcp.client import StdioMcpClient


def _required_env(name: str) -> str:
    value = os.getenv(name)
    assert value, f"{name} must be set for mcp_external tests"
    return value


def _json_env(name: str, default):
    value = os.getenv(name)
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{name} must be valid JSON") from exc


def _load_external_mcp_settings() -> dict:
    command = _required_env("MCP_TEST_SERVER_COMMAND")
    tool_name = _required_env("MCP_TEST_TOOL_NAME")

    args = _json_env("MCP_TEST_SERVER_ARGS_JSON", [])
    assert isinstance(args, list), "MCP_TEST_SERVER_ARGS_JSON must be a JSON array"
    assert all(isinstance(i, str) for i in args), "MCP_TEST_SERVER_ARGS_JSON must contain strings"

    tool_args = _json_env("MCP_TEST_TOOL_ARGUMENTS_JSON", {})
    assert isinstance(tool_args, dict), "MCP_TEST_TOOL_ARGUMENTS_JSON must be a JSON object"

    return {
        "server_name": os.getenv("MCP_TEST_SERVER_NAME", "external"),
        "command": command,
        "args": args,
        "cwd": os.getenv("MCP_TEST_SERVER_CWD"),
        "tool_name": tool_name,
        "tool_args": tool_args,
        "expect_substring": os.getenv("MCP_TEST_EXPECT_SUBSTRING", ""),
    }


@pytest.mark.mcp_external
def test_external_mcp_tool_list_contains_target() -> None:
    cfg = _load_external_mcp_settings()
    client = StdioMcpClient(
        server_name=cfg["server_name"],
        command=cfg["command"],
        args=cfg["args"],
        cwd=cfg["cwd"],
    )
    client.start()
    try:
        tools = client.list_tools()
        names = [t.get("name") for t in tools if isinstance(t, dict)]
        assert cfg["tool_name"] in names, f"tool '{cfg['tool_name']}' not listed by MCP server"
    finally:
        client.close()


@pytest.mark.mcp_external
def test_external_mcp_tool_call() -> None:
    cfg = _load_external_mcp_settings()
    client = StdioMcpClient(
        server_name=cfg["server_name"],
        command=cfg["command"],
        args=cfg["args"],
        cwd=cfg["cwd"],
    )
    client.start()
    try:
        output = client.call_tool(cfg["tool_name"], cfg["tool_args"])
        assert output.strip(), "external MCP tool output is empty"
        if cfg["expect_substring"]:
            assert cfg["expect_substring"] in output
    finally:
        client.close()
