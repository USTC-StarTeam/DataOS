"""Bash tool."""

from __future__ import annotations

import subprocess

from py_agent.tools.base import Tool, ToolError, ToolSpec

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_BYTES = 20_000


def _truncate_output(text: str, limit: int) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    truncated = data[:limit].decode("utf-8", errors="replace")
    return f"{truncated}\n\n[truncated: showing {limit} of {len(data)} bytes]"


def create_bash_tool(cwd: str) -> Tool:
    def _execute(args: dict) -> str:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError("missing required field: command")

        timeout = int(args.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        max_output = int(args.get("max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES))

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"command timed out after {timeout}s") from exc

        stdout = _truncate_output(result.stdout or "", max_output)
        stderr = _truncate_output(result.stderr or "", max_output)

        body = [f"exit_code={result.returncode}"]
        if stdout:
            body.append("[stdout]\n" + stdout)
        if stderr:
            body.append("[stderr]\n" + stderr)

        return "\n\n".join(body)

    return Tool(
        spec=ToolSpec(
            name="bash",
            description="Run a shell command in the workspace",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                    "max_output_bytes": {"type": "integer", "minimum": 1},
                },
                "required": ["command"],
            },
        ),
        execute=_execute,
    )
