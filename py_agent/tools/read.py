"""Read tool."""

from __future__ import annotations

from py_agent.tools.base import Tool, ToolError, ToolSpec
from py_agent.tools.path_utils import resolve_workspace_path

DEFAULT_MAX_BYTES = 20_000


def create_read_tool(cwd: str) -> Tool:
    def _execute(args: dict) -> str:
        path_arg = str(args.get("path", "")).strip()
        if not path_arg:
            raise ToolError("missing required field: path")
        max_bytes = int(args.get("max_bytes", DEFAULT_MAX_BYTES))

        path = resolve_workspace_path(cwd, path_arg)
        if not path.exists() or not path.is_file():
            raise ToolError(f"file not found: {path_arg}")

        content = path.read_bytes()
        if len(content) <= max_bytes:
            return content.decode("utf-8", errors="replace")

        head = content[:max_bytes].decode("utf-8", errors="replace")
        return f"{head}\n\n[truncated: showing {max_bytes} of {len(content)} bytes]"

    return Tool(
        spec=ToolSpec(
            name="read",
            description="Read a file from the workspace",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
            },
        ),
        execute=_execute,
    )
