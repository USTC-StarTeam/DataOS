"""Path safety helpers for workspace-local tools."""

from __future__ import annotations

from pathlib import Path

from py_agent.tools.base import ToolError


def resolve_workspace_path(cwd: str, target: str) -> Path:
    base = Path(cwd).resolve()
    path = (base / target).resolve()
    if base == path or base in path.parents:
        return path
    raise ToolError(f"path escapes workspace: {target}")
