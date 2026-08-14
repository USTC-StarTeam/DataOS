"""把杂乱的函数调用样本规范化成评测器期望的工具调用格式。"""

from __future__ import annotations

import ast
import json
from typing import Any

from dataos_operator.core import MapperOperator, Record

_MISSING = object()

#: python 类型名 -> JSON-schema 类型名的默认映射。
DEFAULT_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "list": "array",
    "dict": "object",
    "float": "number",
    "bool": "boolean",
}


class FunctionCallNormalizer(MapperOperator):
    """Normalize messy tool-call samples into an evaluator-ready messages/tools schema.

    Implementation: read ``tools_key`` (list/dict/JSON string) and rewrite every tool into a
    JSON-schema ``{"type": "object", "properties": ..., "required": ...}`` — parameter types are
    mapped through ``type_map`` and any parameter without a ``default`` becomes required; read
    ``call_key`` (list, JSON string, or ``<functioncall>``-tagged blob) and parse it loosely into
    clean ``[{"name", "arguments": object}]`` calls (single-quoted args fall back to
    ``ast.literal_eval``); then assemble ``messages`` with the tools JSON in a system turn, the
    ``query_key`` text in a user turn, and the calls wrapped in ``<tool_call_tag>`` markers in an
    assistant turn. Field names are parameterized and parsing is content-based, so no dataset
    layout is hard-coded. Missing pieces are skipped rather than filled with blanks.
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "function_call_normalizer"

    def __init__(
        self,
        tools_key: str = "tools",
        call_key: str = "answer",
        query_key: str = "query",
        *,
        tool_call_tag: str = "tool_call",
        messages_key: str = "messages",
        type_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tools_key=tools_key,
            call_key=call_key,
            query_key=query_key,
            tool_call_tag=tool_call_tag,
            messages_key=messages_key,
            type_map=type_map,
            **kwargs,
        )
        self.tools_key = tools_key
        self.call_key = call_key
        self.query_key = query_key
        self.tool_call_tag = tool_call_tag
        self.messages_key = messages_key
        self.type_map = dict(DEFAULT_TYPE_MAP if type_map is None else type_map)

    def map_record(self, record: Record) -> Record:
        result = {
            key: value
            for key, value in record.items()
            if key not in (self.tools_key, self.call_key, self.query_key)
        }

        tools = self._normalize_tools(self.get_field(record, self.tools_key, _MISSING))
        calls = self._parse_calls(self.get_field(record, self.call_key, _MISSING))
        query = self.get_field(record, self.query_key, _MISSING)

        messages: list[dict[str, str]] = []
        if tools:
            messages.append(
                {"role": "system", "content": json.dumps(tools, ensure_ascii=False)}
            )
        if query is not _MISSING and query is not None:
            messages.append({"role": "user", "content": _as_text(query)})
        if calls:
            messages.append({"role": "assistant", "content": self._render_calls(calls)})

        if tools:
            result[self.tools_key] = tools
        result[self.messages_key] = messages
        return result

    # -- tools ---------------------------------------------------------------

    def _normalize_tools(self, raw: Any) -> list[dict[str, Any]]:
        if raw is _MISSING or raw is None:
            return []
        if isinstance(raw, str):
            raw = _try_json(raw)
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [tool for tool in (self._normalize_tool(item) for item in raw) if tool]

    def _normalize_tool(self, tool: Any) -> dict[str, Any] | None:
        if not isinstance(tool, dict):
            return None
        # OpenAI-style {"type": "function", "function": {...}} wrapper.
        if isinstance(tool.get("function"), dict):
            tool = tool["function"]
        name = tool.get("name")
        if name is None:
            return None
        raw_params = tool.get("parameters", tool.get("arguments", {}))
        if isinstance(raw_params, dict) and "properties" in raw_params:
            source_props = raw_params.get("properties", {})
        elif isinstance(raw_params, dict):
            source_props = raw_params
        else:
            source_props = {}

        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param_spec in source_props.items():
            if isinstance(param_spec, dict):
                normalized = dict(param_spec)
                if "type" in normalized:
                    normalized["type"] = self._normalize_type(normalized["type"])
                has_default = "default" in normalized
            else:
                normalized = {"type": self._normalize_type(param_spec)}
                has_default = False
            properties[param_name] = normalized
            if not has_default:
                required.append(param_name)

        normalized_tool: dict[str, Any] = {"name": str(name)}
        description = tool.get("description")
        if description is not None:
            normalized_tool["description"] = _as_text(description)
        normalized_tool["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        return normalized_tool

    def _normalize_type(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return self.type_map.get(value.strip().lower(), value)

    # -- calls ---------------------------------------------------------------

    def _parse_calls(self, raw: Any) -> list[dict[str, Any]]:
        if raw is _MISSING or raw is None:
            return []
        if isinstance(raw, dict):
            items: list[Any] = [raw]
        elif isinstance(raw, list):
            items = list(raw)
        elif isinstance(raw, str):
            items = _parse_call_string(raw)
        else:
            return []
        return [call for call in (_coerce_call(item) for item in items) if call]

    def _render_calls(self, calls: list[dict[str, Any]]) -> str:
        open_tag, close_tag = f"<{self.tool_call_tag}>", f"</{self.tool_call_tag}>"
        return "\n".join(
            f"{open_tag}\n{json.dumps(call, ensure_ascii=False)}\n{close_tag}" for call in calls
        )


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _try_loose(text: str) -> Any:
    """Parse JSON first, then fall back to python-literal (single-quoted) parsing."""

    parsed = _try_json(text)
    if parsed is not None:
        return parsed
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _parse_call_string(text: str) -> list[Any]:
    cleaned = text
    for tag in ("<functioncall>", "</functioncall>", "<tool_call>", "</tool_call>"):
        cleaned = cleaned.replace(tag, " ")
    cleaned = cleaned.strip()
    parsed = _try_loose(cleaned)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return _find_json_objects(cleaned)


def _find_json_objects(text: str) -> list[Any]:
    objects: list[Any] = []
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                parsed = _try_loose(text[start : index + 1])
                if isinstance(parsed, dict):
                    objects.append(parsed)
                start = -1
    return objects


def _coerce_call(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("function"), dict):
        item = item["function"]
    name = item.get("name")
    if name is None:
        return None
    raw_args = item.get("arguments", item.get("parameters"))
    return {"name": str(name), "arguments": _coerce_args(raw_args)}


def _coerce_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = _try_loose(raw)
        if isinstance(parsed, dict):
            return parsed
    return {}
