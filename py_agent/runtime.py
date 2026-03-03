"""Agent runtime loop with local tool execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from py_agent.providers.base import Provider, ProviderError, ToolCall
from py_agent.tools import ToolError

MAX_TOOL_LOOP_STEPS = 8


@dataclass(frozen=True)
class RuntimeResult:
    model: str
    text: str


def build_tool_schemas(tools: dict[str, Any]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in tools.values():
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.spec.name,
                    "description": tool.spec.description,
                    "parameters": tool.spec.parameters,
                },
            }
        )
    return schemas


def run_agent_loop(provider: Provider, model: str, prompt: str, tools: dict[str, Any]) -> RuntimeResult:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_schemas = build_tool_schemas(tools)

    for _ in range(MAX_TOOL_LOOP_STEPS):
        response = provider.chat(messages=messages, model=model, tools=tool_schemas)
        if not response.tool_calls:
            return RuntimeResult(model=response.model, text=response.text)

        messages.append(_assistant_message_for_tool_calls(response.text, response.tool_calls))
        for call in response.tool_calls:
            output = _execute_tool_call(call, tools)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output,
                }
            )

    raise ProviderError(f"tool loop exceeded max steps: {MAX_TOOL_LOOP_STEPS}")


def _assistant_message_for_tool_calls(text: str, tool_calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in tool_calls
        ],
    }


def _execute_tool_call(call: ToolCall, tools: dict[str, Any]) -> str:
    tool = tools.get(call.name)
    if tool is None:
        return f"tool error: unknown tool '{call.name}'"
    try:
        return tool.execute(call.arguments)
    except ToolError as exc:
        return f"tool error: {exc}"
