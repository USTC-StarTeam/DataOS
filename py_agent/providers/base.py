"""Provider abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletionResult:
    model: str
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResult:
    model: str
    text: str
    tool_calls: list[ToolCall]
    assistant_message: dict[str, Any] | None = None


class ProviderError(RuntimeError):
    """Raised when provider request fails."""


class Provider:
    def chat(self, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]] | None = None) -> ChatResult:
        raise NotImplementedError

    def complete(self, prompt: str, model: str) -> CompletionResult:
        result = self.chat(messages=[{"role": "user", "content": prompt}], model=model, tools=None)
        return CompletionResult(model=result.model, text=result.text)
