"""Write tool."""

from __future__ import annotations

from py_agent.tools.base import Tool, ToolError, ToolSpec
from py_agent.tools.path_utils import resolve_workspace_path


def create_write_tool(cwd: str) -> Tool:
    def _execute(args: dict) -> str:
        path_arg = str(args.get("path", "")).strip()
        if not path_arg:
            raise ToolError("missing required field: path")
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolError("missing required field: content")

        path = resolve_workspace_path(cwd, path_arg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {len(content.encode('utf-8'))} bytes to {path_arg}"

    return Tool(
        spec=ToolSpec(
            name="write",
            description="Write content to a file in the workspace",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        execute=_execute,
    )
