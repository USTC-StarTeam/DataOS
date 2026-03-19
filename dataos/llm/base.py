"""Unified LLM abstractions for DataOS operators and runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """Provider-agnostic chat message used by DataOS operators."""

    role: str
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.role, "role")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if self.name is not None:
            _require_text(self.name, "name")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting metadata returned by an LLM provider."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class GenerateRequest:
    """Normalized request payload consumed by the unified LLM interface."""

    model: str
    messages: list[LLMMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.model, "model")
        self.messages = list(self.messages)
        if not self.messages:
            raise ValueError("messages must not be empty")
        self.stop = list(self.stop)
        self.metadata = dict(self.metadata)
        self.extra = dict(self.extra)


@dataclass(slots=True)
class GenerateResponse:
    """Normalized response emitted by all LLM provider adapters."""

    provider: str
    model: str
    text: str
    usage: TokenUsage | None = None
    latency_ms: float | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.provider, "provider")
        _require_text(self.model, "model")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        self.metadata = dict(self.metadata)
        self.raw = dict(self.raw)


class LLMError(RuntimeError):
    """Raised when an LLM provider request fails."""


@runtime_checkable
class LLM(Protocol):
    """Stable provider interface consumed by DataOS operators."""

    provider_name: str

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate a response from a normalized provider-agnostic request."""


def build_messages(prompt: str, *, system_prompt: str | None = None) -> list[LLMMessage]:
    """Create a minimal system+user message list for text generation tasks."""

    _require_text(prompt, "prompt")
    messages: list[LLMMessage] = []
    if system_prompt is not None:
        _require_text(system_prompt, "system_prompt")
        messages.append(LLMMessage(role="system", content=system_prompt))
    messages.append(LLMMessage(role="user", content=prompt))
    return messages


__all__ = [
    "GenerateRequest",
    "GenerateResponse",
    "LLM",
    "LLMError",
    "LLMMessage",
    "TokenUsage",
    "build_messages",
]
