"""把常见 record 或 chat 样本映射为 Alpaca 训练字段。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from dataos_operator.core import MapperOperator, Record, get_field
from dataos_operator.core.chat_formats import (
    coerce_text,
    messages_to_simple_turns,
    normalize_chat_messages,
    normalize_history_pairs,
)

SourceFormat = Literal["auto", "fielded", "messages"]
_MISSING = object()


class AlpacaMapper(MapperOperator):
    """Map fielded or message samples into the Alpaca schema (instruction/input/output/system/history).

    Implementation: detect by ``source_format`` (auto/fielded/messages)—fielded samples copy
    configured fields directly; message samples split ``messages`` into turns, with the last
    turn as instruction/output and earlier turns as history. Consumed source fields are removed
    from the result; missing fields are skipped, not filled with blanks.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        source_format: SourceFormat = "auto",
        instruction_key: str = "text",
        input_key: str = "query",
        output_key: str = "answer",
        system_key: str = "system",
        history_key: str = "history",
        messages_key: str = "messages",
        strict: bool = True,
        always_include_input: bool = False,
        default_input: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_format=source_format,
            instruction_key=instruction_key,
            input_key=input_key,
            output_key=output_key,
            system_key=system_key,
            history_key=history_key,
            messages_key=messages_key,
            strict=strict,
            always_include_input=always_include_input,
            default_input=default_input,
            **kwargs,
        )
        self.source_format = source_format
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.system_key = system_key
        self.history_key = history_key
        self.messages_key = messages_key
        self.strict = strict
        self.always_include_input = always_include_input
        self.default_input = coerce_text(default_input)

    def map_record(self, record: Record) -> Record:
        resolved_source_format = self._resolved_source_format(record)
        mapped_record = _drop_source_fields(record, self._source_keys_for(resolved_source_format))
        if resolved_source_format == "messages":
            target_fields = self._map_from_messages(record)
        else:
            target_fields = self._map_from_fields(record)

        mapped_record.update(target_fields)
        return mapped_record

    def _resolved_source_format(self, record: Record) -> SourceFormat:
        if self.source_format != "auto":
            return self.source_format
        return "messages" if _get_optional(record, self.messages_key)[0] else "fielded"

    def _source_keys_for(self, source_format: SourceFormat) -> tuple[str, ...]:
        if source_format == "messages":
            return (self.messages_key, self.system_key)
        return (
            self.instruction_key,
            self.input_key,
            self.output_key,
            self.system_key,
            self.history_key,
        )

    def _map_from_fields(self, record: Record) -> dict[str, Any]:
        target_fields: dict[str, Any] = {}

        instruction_present, instruction_value = _get_optional(record, self.instruction_key)
        if instruction_present:
            target_fields["instruction"] = coerce_text(instruction_value)

        input_present, input_value = _get_optional(record, self.input_key)
        if input_present:
            target_fields["input"] = coerce_text(input_value)
        elif self.always_include_input:
            target_fields["input"] = self.default_input

        output_present, output_value = _get_optional(record, self.output_key)
        if output_present:
            target_fields["output"] = coerce_text(output_value)

        system_present, system_value = _get_optional(record, self.system_key)
        if system_present:
            target_fields["system"] = coerce_text(system_value)

        history_present, history_value = _get_optional(record, self.history_key)
        if history_present:
            history_pairs = normalize_history_pairs(history_value, strict=self.strict)
            target_fields["history"] = [[user, assistant] for user, assistant in history_pairs]

        return target_fields

    def _map_from_messages(self, record: Record) -> dict[str, Any]:
        target_fields: dict[str, Any] = {}
        messages_present, messages_value = _get_optional(record, self.messages_key)
        system_present, system_value = _get_optional(record, self.system_key)

        if not messages_present:
            if system_present:
                target_fields["system"] = coerce_text(system_value)
            return target_fields

        system_from_messages, normalized_messages = normalize_chat_messages(
            messages_value,
            strict=self.strict,
        )
        turns = messages_to_simple_turns(normalized_messages, strict=self.strict)
        history_pairs = turns[:-1]
        current_user, current_assistant = turns[-1]

        target_fields["instruction"] = current_user
        target_fields["output"] = current_assistant
        if self.always_include_input:
            target_fields["input"] = self.default_input
        if history_pairs:
            target_fields["history"] = [[user, assistant] for user, assistant in history_pairs]

        resolved_system = system_value if system_present else system_from_messages
        if resolved_system is not None:
            target_fields["system"] = coerce_text(resolved_system)
        return target_fields


def _get_optional(record: Record, key: str) -> tuple[bool, Any | None]:
    value = get_field(record, key, _MISSING)
    if value is _MISSING or value is None:
        return False, None
    return True, value


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
