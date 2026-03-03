"""Tool abstractions for local tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    execute: Callable[[dict[str, Any]], str]


class ToolError(RuntimeError):
    """Raised for tool execution failures."""
