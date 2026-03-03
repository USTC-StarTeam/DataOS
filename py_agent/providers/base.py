"""Provider abstractions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionResult:
    model: str
    text: str


class ProviderError(RuntimeError):
    """Raised when provider request fails."""


class Provider:
    def complete(self, prompt: str, model: str) -> CompletionResult:
        raise NotImplementedError
