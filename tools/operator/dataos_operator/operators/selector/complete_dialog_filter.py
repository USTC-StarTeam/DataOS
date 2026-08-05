"""只保留含完整非空 assistant 回复的对话样本。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator

_MISSING = object()
#: 视为"助手"角色的常见取值(兼容 messages / ShareGPT 等格式)。
_ASSISTANT_ROLES = frozenset({"assistant", "gpt", "bot", "model"})


class CompleteDialogFilter(SelectorOperator):
    """Keep only dialogues that contain a complete, non-empty assistant reply.

    Implementation: read the ``messages_key`` conversation and keep the row when at least one turn
    whose ``role_key`` names an assistant (``assistant``/``gpt``/``bot``/``model``) has
    ``content_key`` text of length >= ``min_assistant_chars`` after stripping. All keys are
    parameterized (so ShareGPT ``{from, value}`` turns work by setting ``role_key``/``content_key``)
    and the check is content-based, so no dataset layout is assumed. Use it to drop truncated or
    empty-response chats before training on a conversational benchmark (e.g. HealthBench, Arena).
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "complete_dialog_filter"

    def __init__(
        self,
        *,
        messages_key: str = "messages",
        min_assistant_chars: int = 1,
        role_key: str = "role",
        content_key: str = "content",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            messages_key=messages_key,
            min_assistant_chars=min_assistant_chars,
            role_key=role_key,
            content_key=content_key,
            **kwargs,
        )
        self.messages_key = messages_key
        self.min_assistant_chars = min_assistant_chars
        self.role_key = role_key
        self.content_key = content_key

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                if self._has_complete_reply(record):
                    selected.append(record)
        return selected

    def _has_complete_reply(self, record: Record) -> bool:
        messages = self.get_field(record, self.messages_key, _MISSING)
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            return False
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get(self.role_key)
            if not isinstance(role, str) or role.strip().lower() not in _ASSISTANT_ROLES:
                continue
            content = message.get(self.content_key)
            if content is None:
                continue
            if len(str(content).strip()) >= self.min_assistant_chars:
                return True
        return False
