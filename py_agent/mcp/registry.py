"""MCP registry that exposes remote tools as local Tool objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from py_agent.mcp.adapters import normalize_tool_schema
from py_agent.mcp.client import McpClientError, StdioMcpClient
from py_agent.tools.base import Tool, ToolError, ToolSpec


class McpRegistryError(RuntimeError):
    """Raised when MCP registry setup fails."""


class McpRegistry:
    def __init__(self, config_path: Path, enabled_servers: list[str] | None = None) -> None:
        self.config_path = Path(config_path)
        self.enabled_servers = list(enabled_servers or [])
        self._clients: dict[str, StdioMcpClient] = {}
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        config = _load_config(self.config_path)
        servers = config.get("servers")
        if not isinstance(servers, dict):
            raise McpRegistryError("invalid MCP config: 'servers' must be an object")

        selected = self.enabled_servers or list(servers.keys())
        if not selected:
            raise McpRegistryError("MCP config has no servers")

        clients: dict[str, StdioMcpClient] = {}
        for name in selected:
            if name not in servers:
                raise McpRegistryError(f"MCP server not found in config: {name}")
            entry = servers[name]
            if not isinstance(entry, dict):
                raise McpRegistryError(f"invalid MCP server entry: {name}")
            command = entry.get("command")
            if not isinstance(command, str) or not command:
                raise McpRegistryError(f"invalid MCP command for server: {name}")
            args = entry.get("args", [])
            if not isinstance(args, list) or not all(isinstance(i, str) for i in args):
                raise McpRegistryError(f"invalid MCP args for server: {name}")
            env = entry.get("env", {})
            if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                raise McpRegistryError(f"invalid MCP env for server: {name}")
            cwd = entry.get("cwd")
            if cwd is not None and not isinstance(cwd, str):
                raise McpRegistryError(f"invalid MCP cwd for server: {name}")
            client = StdioMcpClient(
                server_name=name,
                command=command,
                args=args,
                env=env,
                cwd=cwd,
            )
            client.start()
            clients[name] = client

        self._clients = clients
        self._started = True

    def build_tools(self) -> dict[str, Tool]:
        if not self._started:
            raise McpRegistryError("MCP registry is not started")
        merged: dict[str, Tool] = {}
        for server_name, client in self._clients.items():
            try:
                tool_defs = client.list_tools()
            except McpClientError as exc:
                raise McpRegistryError(str(exc)) from exc
            for tool_def in tool_defs:
                raw_name = tool_def.get("name")
                if not isinstance(raw_name, str) or not raw_name:
                    continue
                namespaced_name = f"mcp.{server_name}.{raw_name}"
                if namespaced_name in merged:
                    raise McpRegistryError(f"duplicate MCP tool: {namespaced_name}")
                description = tool_def.get("description")
                if not isinstance(description, str):
                    description = f"MCP tool {raw_name} from server {server_name}"
                input_schema = normalize_tool_schema(tool_def)
                merged[namespaced_name] = Tool(
                    spec=ToolSpec(
                        name=namespaced_name,
                        description=description,
                        parameters=input_schema,
                    ),
                    execute=_build_executor(client, raw_name),
                )
        return merged

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
        self._started = False


def _build_executor(client: StdioMcpClient, tool_name: str):
    def _execute(arguments: dict[str, Any]) -> str:
        try:
            return client.call_tool(tool_name, arguments)
        except McpClientError as exc:
            raise ToolError(str(exc)) from exc

    return _execute


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise McpRegistryError(f"MCP config not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpRegistryError(f"invalid MCP config JSON: {path}") from exc
