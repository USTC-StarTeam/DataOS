"""Map paired user/assistant turns to the standard messages schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core import MapperOperator, Record


class PairedConversationMapper(MapperOperator):
    """Map ``[{user, assistant}, ...]`` turns to messages without dropping source fields.

    Implementation: emit one user and one assistant message for each paired turn, preserving the
    original ``texts`` and image payloads by default; strict mode rejects malformed/empty turns.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        *,
        texts_key: str = "texts",
        messages_key: str = "messages",
        user_key: str = "user",
        assistant_key: str = "assistant",
        preserve_texts: bool = True,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            texts_key=texts_key,
            messages_key=messages_key,
            user_key=user_key,
            assistant_key=assistant_key,
            preserve_texts=preserve_texts,
            strict=strict,
            **kwargs,
        )
        self.texts_key = texts_key
        self.messages_key = messages_key
        self.user_key = user_key
        self.assistant_key = assistant_key
        self.preserve_texts = preserve_texts
        self.strict = strict

    def map_record(self, record: Record) -> Record:
        raw_turns = self.require_field(record, self.texts_key)
        if not isinstance(raw_turns, Sequence) or isinstance(raw_turns, str | bytes):
            raise ValueError("paired conversation must be a sequence")
        messages: list[dict[str, str]] = []
        for turn in raw_turns:
            if not isinstance(turn, Mapping):
                if self.strict:
                    raise ValueError("each paired turn must be a mapping")
                continue
            user = turn.get(self.user_key)
            assistant = turn.get(self.assistant_key)
            if user is None or assistant is None:
                if self.strict:
                    raise ValueError("paired turn must contain user and assistant fields")
                continue
            user_text = str(user).strip()
            assistant_text = str(assistant).strip()
            if self.strict and (not user_text or not assistant_text):
                raise ValueError("paired turn contents must not be empty")
            messages.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
        if self.strict and not messages:
            raise ValueError("paired conversation contains no usable turns")
        updated = dict(self.set_field(record, self.messages_key, messages))
        if not self.preserve_texts and "." not in self.texts_key:
            updated.pop(self.texts_key, None)
        return updated
