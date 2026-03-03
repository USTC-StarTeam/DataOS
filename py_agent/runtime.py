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
    turn_messages: list[dict[str, Any]]
    events: list[dict[str, Any]]


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


def run_agent_loop(
    provider: Provider,
    model: str,
    prompt: str,
    tools: dict[str, Any],
    initial_messages: list[dict[str, Any]] | None = None,
) -> RuntimeResult:
    messages: list[dict[str, Any]] = list(initial_messages or [])
    events: list[dict[str, Any]] = []
    start_index = len(messages)
    messages.append({"role": "user", "content": prompt})
    events.append({"type": "user", "content": prompt})
    tool_schemas = build_tool_schemas(tools)
    latest_tool_outputs: list[str] = []

    for _ in range(MAX_TOOL_LOOP_STEPS):
        try:
            response = provider.chat(messages=messages, model=model, tools=tool_schemas)
        except ProviderError as exc:
            # Some OpenAI-compatible gateways reject round-trip tool messages.
            # Fall back to returning tool execution output if tools already ran.
            exc_text = str(exc).lower()
            if latest_tool_outputs and "messages with role" in exc_text and "tool" in exc_text:
                text = "\n".join(latest_tool_outputs)
                messages.append({"role": "assistant", "content": text})
                events.append({"type": "assistant", "content": text, "fallback": True})
                return RuntimeResult(model=model, text=text, turn_messages=messages[start_index:], events=events)
            raise
        if not response.tool_calls:
            messages.append({"role": "assistant", "content": response.text})
            events.append({"type": "assistant", "content": response.text})
            return RuntimeResult(
                model=response.model,
                text=response.text,
                turn_messages=messages[start_index:],
                events=events,
            )

        if response.assistant_message is not None:
            messages.append(response.assistant_message)
        else:
            messages.append(_assistant_message_for_tool_calls(response.text, response.tool_calls))
        for call in response.tool_calls:
            events.append(
                {
                    "type": "tool_call",
                    "tool": call.name,
                    "tool_call_id": call.id,
                    "arguments": call.arguments,
                }
            )
        for call in response.tool_calls:
            output = _execute_tool_call(call, tools)
            latest_tool_outputs.append(output)
            messages.append(
                {
                    "role": "tool",
                    "name": call.name,
                    "tool_call_id": call.id,
                    "content": output,
                }
            )
            events.append(
                {
                    "type": "tool_result",
                    "tool": call.name,
                    "tool_call_id": call.id,
                    "content": output,
                }
            )

    raise ProviderError(f"tool loop exceeded max steps: {MAX_TOOL_LOOP_STEPS}")


def _assistant_message_for_tool_calls(text: str, tool_calls: list[ToolCall]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
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
    # Some OpenAI-compatible gateways require null/omitted content when assistant emits tool_calls.
    message["content"] = text if text else None
    return message


def _execute_tool_call(call: ToolCall, tools: dict[str, Any]) -> str:
    tool = tools.get(call.name)
    if tool is None:
        return f"tool error: unknown tool '{call.name}'"
    try:
        return tool.execute(call.arguments)
    except ToolError as exc:
        return f"tool error: {exc}"
