"""Built-in local tools."""

from __future__ import annotations

from py_agent.tools.base import Tool, ToolError
from py_agent.tools.bash import create_bash_tool
from py_agent.tools.edit import create_edit_tool
from py_agent.tools.read import create_read_tool
from py_agent.tools.write import create_write_tool


def create_local_tools(cwd: str) -> dict[str, Tool]:
    tools = [
        create_read_tool(cwd),
        create_write_tool(cwd),
        create_edit_tool(cwd),
        create_bash_tool(cwd),
    ]
    return {tool.spec.name: tool for tool in tools}


__all__ = ["Tool", "ToolError", "create_local_tools"]
