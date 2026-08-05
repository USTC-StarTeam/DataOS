"""聊天与训练数据格式转换的共享工具。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core.contracts import get_field

_SIMPLE_TURN_ROLES = {"user", "assistant"}
_SHAREGPT_ODD_ROLES = {"human", "observation"}
_SHAREGPT_EVEN_ROLES = {"gpt", "function_call"}


def coerce_text(value: Any) -> str:
    """把任意值稳定地转换成字符串。"""

    if value is None:
        return ""
    return str(value)


LLAMAFACTORY_ALPACA_PROMPT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request."
)


def format_llamafactory_alpaca(
    instruction: str,
    *,
    extra_input: str | None = None,
    response: str | None = None,
) -> str:
    """按 LlamaFactory 的 alpaca 风格拼接指令与可选回答。"""

    instruction = coerce_text(instruction).strip()
    extra_input = coerce_text(extra_input).strip() if extra_input is not None else ""
    if not instruction and not extra_input:
        return ""

    instruction_block = instruction
    if extra_input:
        instruction_block = "\n".join(part for part in (instruction, extra_input) if part)

    prompt = (
        f"{LLAMAFACTORY_ALPACA_PROMPT}\n\n"
        f"### Instruction:\n{instruction_block}\n\n"
        "### Response:"
    )
    if response is None:
        return prompt
    return f"{prompt}\n{coerce_text(response).strip()}"


def maybe_get_field(record: Mapping[str, Any], key: str) -> Any | None:
    """读取可选字段，缺失时返回 None。"""

    value = get_field(record, key, None)
    return value


def require_string_field(record: Mapping[str, Any], key: str) -> str:
    """读取必需字段并转成字符串。"""

    value = get_field(record, key, None)
    if value is None:
        raise KeyError(f"Required field is missing: {key}")
    return coerce_text(value)


def normalize_history_pairs(value: Any, *, strict: bool) -> list[tuple[str, str]]:
    """把历史对话标准化为 (user, assistant) pair 列表。"""

    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        if strict:
            raise ValueError("history must be a sequence of pairs")
        return []

    pairs: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, Mapping):
            user_value = _first_present(item, "user", "human", "instruction", "prompt")
            assistant_value = _first_present(item, "assistant", "gpt", "output", "response")
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            if len(item) != 2:
                if strict:
                    raise ValueError("history pair sequence must contain exactly two items")
                continue
            user_value, assistant_value = item
        else:
            if strict:
                raise ValueError("history item must be a pair mapping or a two-item sequence")
            continue

        if user_value is None or assistant_value is None:
            if strict:
                raise ValueError("history pair must contain both user and assistant content")
            continue
        pairs.append((coerce_text(user_value), coerce_text(assistant_value)))
    return pairs


def normalize_chat_messages(value: Any, *, strict: bool) -> tuple[str | None, list[dict[str, str]]]:
    """把多种常见 message 结构标准化为 role/content 形式。"""

    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("messages must be a sequence of message objects")

    system_message: str | None = None
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            if strict:
                raise ValueError("each message must be a mapping")
            continue

        role_value = _first_present(item, "role", "from")
        if role_value is None:
            if strict:
                raise ValueError("message role is missing")
            continue

        role = _normalize_role(coerce_text(role_value))
        if role is None:
            if strict:
                raise ValueError(f"unsupported message role: {role_value}")
            continue

        content_value = _first_present(item, "content", "value")
        if content_value is None:
            if strict:
                raise ValueError("message content is missing")
            continue
        content = coerce_text(content_value)

        if role == "system":
            if system_message is not None and strict:
                raise ValueError("multiple system messages are not supported")
            if system_message is None:
                system_message = content
            continue

        normalized.append({"role": role, "content": content})
    return system_message, normalized


def messages_to_simple_turns(
    messages: Sequence[Mapping[str, str]],
    *,
    strict: bool,
) -> list[tuple[str, str]]:
    """把简单 user/assistant 消息流转换成 pair 列表。"""

    turns: list[tuple[str, str]] = []
    pending_user: str | None = None

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role not in _SIMPLE_TURN_ROLES:
            if strict:
                raise ValueError(f"unsupported role for Alpaca mapping: {role}")
            continue

        if role == "user":
            if pending_user is not None:
                raise ValueError("messages must alternate between user and assistant")
            pending_user = content
            continue

        if pending_user is None:
            raise ValueError("assistant message must follow a user message")
        turns.append((pending_user, content))
        pending_user = None

    if pending_user is not None:
        raise ValueError("messages end with an incomplete user turn")
    if not turns:
        raise ValueError("messages must contain at least one complete user/assistant turn")
    return turns


def messages_to_sharegpt_conversations(
    messages: Sequence[Mapping[str, str]],
    *,
    strict: bool,
) -> list[dict[str, str]]:
    """把标准化消息流转换为 ShareGPT conversations。"""

    conversations: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        target_role = _sharegpt_role(role)
        if target_role is None:
            if strict:
                raise ValueError(f"unsupported role for ShareGPT mapping: {role}")
            continue
        conversations.append({"from": target_role, "value": message["content"]})

    validate_sharegpt_conversations(conversations, strict=strict)
    return conversations


def validate_sharegpt_conversations(
    conversations: Sequence[Mapping[str, str]],
    *,
    strict: bool,
) -> None:
    """校验 ShareGPT conversations 的角色交替是否合法。"""

    if not conversations:
        raise ValueError("conversations must contain at least one message")

    for index, message in enumerate(conversations):
        role = message.get("from")
        if role is None:
            raise ValueError("ShareGPT conversation message is missing from")

        expected_roles = (
            _SHAREGPT_ODD_ROLES if index % 2 == 0 else _SHAREGPT_EVEN_ROLES
        )
        if role not in expected_roles:
            if strict:
                expected_display = "/".join(sorted(expected_roles))
                raise ValueError(
                    f"invalid ShareGPT role order at index {index}: "
                    f"expected {expected_display}, got {role}"
                )


def join_prompt_parts(
    instruction: str | None,
    extra_input: str | None,
    *,
    joiner: str,
    strict: bool,
) -> str:
    """按统一策略拼接主指令和附加输入。"""

    parts = [part for part in [instruction, extra_input] if part]
    if not parts:
        if strict:
            raise ValueError("at least one of instruction or input must be provided")
        return ""
    return joiner.join(parts)


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _normalize_role(role: str) -> str | None:
    lowered = role.strip().lower()
    role_aliases = {
        "system": "system",
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "tool": "observation",
        "observation": "observation",
        "function_call": "function_call",
    }
    return role_aliases.get(lowered)


def _sharegpt_role(role: str) -> str | None:
    role_map = {
        "user": "human",
        "assistant": "gpt",
        "observation": "observation",
        "function_call": "function_call",
    }
    return role_map.get(role)
