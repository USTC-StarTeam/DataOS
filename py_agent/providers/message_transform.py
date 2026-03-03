"""Normalize OpenAI chat messages for tool-call pairing compatibility."""

from __future__ import annotations

from typing import Any


def transform_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure tool messages are paired with a preceding assistant tool_calls message.

    Strategy:
    - Keep all valid user/system/assistant/tool messages.
    - Track pending tool_call ids from assistant.tool_calls.
    - Drop orphan tool messages that cannot be paired.
    - Insert synthetic tool messages for missing tool results before flow continues.
    """

    out: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, str] = {}

    def close_pending() -> None:
        nonlocal pending_tool_calls
        if not pending_tool_calls:
            return
        for call_id, tool_name in pending_tool_calls.items():
            out.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": call_id,
                    "content": "tool error: missing tool result",
                }
            )
        pending_tool_calls = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            close_pending()
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                valid_calls: list[dict[str, Any]] = []
                next_pending: dict[str, str] = {}
                for item in tool_calls:
                    if not isinstance(item, dict):
                        continue
                    call_id = item.get("id")
                    fn = item.get("function")
                    if not isinstance(call_id, str) or not call_id:
                        continue
                    if not isinstance(fn, dict):
                        continue
                    name = fn.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    valid_calls.append(item)
                    next_pending[call_id] = name
                if valid_calls:
                    copied = dict(msg)
                    copied["tool_calls"] = valid_calls
                    out.append(copied)
                    pending_tool_calls = next_pending
                    continue
            # Plain assistant message
            out.append(msg)
            continue

        if role == "tool":
            call_id = msg.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                continue
            if call_id not in pending_tool_calls:
                # Orphan tool message cannot be validated by strict gateways.
                continue
            out.append(msg)
            del pending_tool_calls[call_id]
            continue

        # User/system/others close pending chain first.
        close_pending()
        out.append(msg)

    close_pending()
    return out
