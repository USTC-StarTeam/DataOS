"""Helpers to normalize MCP tool metadata."""

from __future__ import annotations

from typing import Any


def normalize_tool_schema(raw: dict[str, Any]) -> dict[str, Any]:
    schema = raw.get("inputSchema")
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}
