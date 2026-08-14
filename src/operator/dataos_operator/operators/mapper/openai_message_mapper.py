"""把常见 record 或 chat 样本映射为 OpenAI messages 字段。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from dataos_operator.core import MapperOperator, Record, get_field
from dataos_operator.core.chat_formats import (
    coerce_text,
    normalize_chat_messages,
    normalize_history_pairs,
)

SourceFormat = Literal["auto", "alpaca", "qa", "messages", "sharegpt"]
_MISSING = object()


class OpenAIMessageMapper(MapperOperator):
    """Map common SFT samples (alpaca/qa/messages/sharegpt) into the OpenAI messages schema.

    Implementation: with ``source_format=auto`` detect in order messages, conversations, qa,
    then alpaca; fielded samples merge instruction(+input) into a user message and output into
    an assistant message, expanding system/history; role alternation is validated and consumed
    source fields are removed. A clean conversion that adds no extra prompt template.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        source_format: SourceFormat = "auto",
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        query_key: str = "query",
        answer_key: str = "answer",
        system_key: str = "system",
        history_key: str = "history",
        messages_key: str = "messages",
        conversations_key: str = "conversations",
        tools_key: str = "tools",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_format=source_format,
            instruction_key=instruction_key,
            input_key=input_key,
            output_key=output_key,
            query_key=query_key,
            answer_key=answer_key,
            system_key=system_key,
            history_key=history_key,
            messages_key=messages_key,
            conversations_key=conversations_key,
            tools_key=tools_key,
            **kwargs,
        )
        self.source_format = source_format
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.system_key = system_key
        self.history_key = history_key
        self.messages_key = messages_key
        self.conversations_key = conversations_key
        self.tools_key = tools_key

    def map_record(self, record: Record) -> Record:
        resolved_source_format = self._resolved_source_format(record)
        mapped_record = _drop_source_fields(record, self._source_keys_for(resolved_source_format))
        target_fields = self._map_to_messages(record, resolved_source_format)
        mapped_record.update(target_fields)
        return mapped_record

    def _resolved_source_format(self, record: Record) -> SourceFormat:
        if self.source_format != "auto":
            return self.source_format
        if _get_optional(record, self.messages_key)[0]:
            return "messages"
        if _get_optional(record, self.conversations_key)[0]:
            return "sharegpt"
        if _get_optional(record, self.query_key)[0] or _get_optional(record, self.answer_key)[0]:
            return "qa"
        return "alpaca"

    def _source_keys_for(self, source_format: SourceFormat) -> tuple[str, ...]:
        if source_format == "messages":
            return (self.messages_key, self.system_key)
        if source_format == "sharegpt":
            return (self.conversations_key, self.system_key, self.tools_key)
        if source_format == "qa":
            return (
                self.query_key,
                self.answer_key,
                self.system_key,
                self.history_key,
            )
        return (
            self.instruction_key,
            self.input_key,
            self.output_key,
            self.system_key,
            self.history_key,
        )

    def _map_to_messages(self, record: Record, source_format: SourceFormat) -> dict[str, Any]:
        if source_format == "messages":
            return {"messages": self._map_from_messages(record)}
        if source_format == "sharegpt":
            target_fields = {"messages": self._map_from_sharegpt(record)}
            tools_present, tools_value = _get_optional(record, self.tools_key)
            if tools_present:
                target_fields["tools"] = tools_value
            return target_fields
        if source_format == "qa":
            return {"messages": self._map_from_qa(record)}
        return {"messages": self._map_from_alpaca(record)}

    def _map_from_messages(self, record: Record) -> list[dict[str, str]]:
        messages_present, messages_value = _get_optional(record, self.messages_key)
        if not messages_present:
            raise KeyError(f"Required field is missing: {self.messages_key}")

        system_from_messages, normalized_messages = normalize_chat_messages(
            messages_value,
            strict=True,
        )
        output_messages: list[dict[str, str]] = []
        system_present, system_value = _get_optional(record, self.system_key)
        resolved_system = system_value if system_present else system_from_messages
        if resolved_system is not None:
            output_messages.append({"role": "system", "content": coerce_text(resolved_system)})
        output_messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in normalized_messages
        )
        if len(output_messages) == 0:
            raise ValueError("messages must contain at least one usable message")
        return output_messages

    def _map_from_sharegpt(self, record: Record) -> list[dict[str, str]]:
        conversations_present, conversations_value = _get_optional(record, self.conversations_key)
        if not conversations_present:
            raise KeyError(f"Required field is missing: {self.conversations_key}")
        if not isinstance(conversations_value, Sequence) or isinstance(
            conversations_value,
            str | bytes | bytearray,
        ):
            raise ValueError("conversations must be a sequence of mapping items")

        output_messages: list[dict[str, str]] = []
        system_present, system_value = _get_optional(record, self.system_key)
        if system_present:
            output_messages.append({"role": "system", "content": coerce_text(system_value)})

        for item in conversations_value:
            if not isinstance(item, Mapping):
                raise ValueError("each conversation item must be a mapping")
            role_value = item.get("from")
            content_value = item.get("value")
            if role_value is None or content_value is None:
                raise ValueError("conversation item must contain from and value")
            role = _sharegpt_to_openai_role(coerce_text(role_value))
            output_messages.append({"role": role, "content": coerce_text(content_value)})

        _validate_openai_turns(output_messages)
        return output_messages

    def _map_from_qa(self, record: Record) -> list[dict[str, str]]:
        query_present, query_value = _get_optional(record, self.query_key)
        answer_present, answer_value = _get_optional(record, self.answer_key)
        if not query_present:
            raise KeyError(f"Required field is missing: {self.query_key}")
        if not answer_present:
            raise KeyError(f"Required field is missing: {self.answer_key}")

        messages = self._messages_prefix(record)
        history_present, history_value = _get_optional(record, self.history_key)
        if history_present:
            history_pairs = normalize_history_pairs(history_value, strict=True)
            for user, assistant in history_pairs:
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})

        messages.append({"role": "user", "content": coerce_text(query_value)})
        messages.append({"role": "assistant", "content": coerce_text(answer_value)})
        return messages

    def _map_from_alpaca(self, record: Record) -> list[dict[str, str]]:
        instruction_present, instruction_value = _get_optional(record, self.instruction_key)
        output_present, output_value = _get_optional(record, self.output_key)
        if not instruction_present:
            raise KeyError(f"Required field is missing: {self.instruction_key}")
        if not output_present:
            raise KeyError(f"Required field is missing: {self.output_key}")

        messages = self._messages_prefix(record)
        history_present, history_value = _get_optional(record, self.history_key)
        if history_present:
            history_pairs = normalize_history_pairs(history_value, strict=True)
            for user, assistant in history_pairs:
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})

        input_present, input_value = _get_optional(record, self.input_key)
        user_content = _merge_instruction_and_input(
            coerce_text(instruction_value),
            coerce_text(input_value) if input_present else None,
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": coerce_text(output_value)})
        return messages

    def _messages_prefix(self, record: Record) -> list[dict[str, str]]:
        system_present, system_value = _get_optional(record, self.system_key)
        if not system_present:
            return []
        return [{"role": "system", "content": coerce_text(system_value)}]


def _get_optional(record: Record, key: str) -> tuple[bool, Any | None]:
    value = get_field(record, key, _MISSING)
    if value is _MISSING or value is None:
        return False, None
    return True, value


def _merge_instruction_and_input(instruction: str, extra_input: str | None) -> str:
    instruction = instruction.strip()
    normalized_input = extra_input.strip() if extra_input is not None else ""
    if not normalized_input:
        return instruction
    if not instruction:
        return normalized_input
    return f"{instruction}\n{normalized_input}"


def _sharegpt_to_openai_role(role: str) -> str:
    lowered = role.strip().lower()
    if lowered in {"human", "user", "observation"}:
        return "user"
    if lowered in {"gpt", "assistant", "function_call"}:
        return "assistant"
    if lowered == "system":
        return "system"
    raise ValueError(f"unsupported ShareGPT role: {role}")


def _validate_openai_turns(messages: Sequence[Mapping[str, str]]) -> None:
    non_system_messages = [message for message in messages if message["role"] != "system"]
    if not non_system_messages:
        raise ValueError("messages must contain at least one non-system message")

    expected_role = "user"
    for message in non_system_messages:
        role = message["role"]
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported OpenAI role: {role}")
        if role != expected_role:
            raise ValueError(f"messages must alternate between user and assistant, got {role}")
        expected_role = "assistant" if expected_role == "user" else "user"


def _drop_source_fields(record: Record, keys: Sequence[str]) -> dict[str, Any]:
    updated: Any = dict(record)
    for key in dict.fromkeys(keys):
        updated, _, should_prune = _drop_field_path(updated, key.split("."))
        if should_prune:
            updated = {}
    return dict(updated)


def _drop_field_path(current: Any, parts: Sequence[str]) -> tuple[Any, bool, bool]:
    part = parts[0]
    is_leaf = len(parts) == 1

    if isinstance(current, Mapping):
        if part not in current:
            return current, False, False

        updated_mapping = dict(current)
        if is_leaf:
            del updated_mapping[part]
        else:
            child, removed, should_prune_child = _drop_field_path(updated_mapping[part], parts[1:])
            if not removed:
                return current, False, False
            if should_prune_child:
                del updated_mapping[part]
            else:
                updated_mapping[part] = child
        return updated_mapping, True, len(updated_mapping) == 0

    if isinstance(current, Sequence) and not isinstance(current, str | bytes | bytearray):
        if not part.isdigit():
            return current, False, False

        index = int(part)
        if index >= len(current):
            return current, False, False

        updated_list = list(current)
        if is_leaf:
            updated_list.pop(index)
        else:
            child, removed, should_prune_child = _drop_field_path(updated_list[index], parts[1:])
            if not removed:
                return current, False, False
            if should_prune_child:
                updated_list.pop(index)
            else:
                updated_list[index] = child
        return updated_list, True, len(updated_list) == 0

    return current, False, False
