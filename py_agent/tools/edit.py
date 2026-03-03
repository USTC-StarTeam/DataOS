"""Edit tool."""

from __future__ import annotations

from py_agent.tools.base import Tool, ToolError, ToolSpec
from py_agent.tools.path_utils import resolve_workspace_path


def create_edit_tool(cwd: str) -> Tool:
    def _execute(args: dict) -> str:
        path_arg = str(args.get("path", "")).strip()
        old_text = args.get("old_text")
        new_text = args.get("new_text")
        replace_all = bool(args.get("replace_all", False))

        if not path_arg:
            raise ToolError("missing required field: path")
        if not isinstance(old_text, str):
            raise ToolError("missing required field: old_text")
        if not isinstance(new_text, str):
            raise ToolError("missing required field: new_text")

        path = resolve_workspace_path(cwd, path_arg)
        if not path.exists() or not path.is_file():
            raise ToolError(f"file not found: {path_arg}")

        original = path.read_text(encoding="utf-8")
        count = original.count(old_text)
        if count == 0:
            raise ToolError("old_text not found")

        if replace_all:
            updated = original.replace(old_text, new_text)
            replaced = count
        else:
            updated = original.replace(old_text, new_text, 1)
            replaced = 1

        path.write_text(updated, encoding="utf-8")
        return f"edited {path_arg}: replaced {replaced} occurrence(s)"

    return Tool(
        spec=ToolSpec(
            name="edit",
            description="Edit a file by replacing old_text with new_text",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        ),
        execute=_execute,
    )
