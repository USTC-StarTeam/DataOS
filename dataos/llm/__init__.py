"""LLM abstraction layer for DataOS."""

from __future__ import annotations

from typing import Any

from .adapters import OpenAICompatibleLLM, VLLMOpenAIAdapter
from .base import GenerateRequest, GenerateResponse, LLM, LLMError, LLMMessage, TokenUsage, build_messages


def create_llm(provider: str, **kwargs: Any) -> LLM:
    """Instantiate a concrete LLM adapter behind the unified DataOS interface."""

    normalized = provider.strip().lower()
    if normalized in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleLLM(**kwargs)
    if normalized == "vllm":
        return VLLMOpenAIAdapter(**kwargs)
    raise ValueError(f"unsupported provider: {provider}")


__all__ = [
    "GenerateRequest",
    "GenerateResponse",
    "LLM",
    "LLMError",
    "LLMMessage",
    "OpenAICompatibleLLM",
    "TokenUsage",
    "VLLMOpenAIAdapter",
    "build_messages",
    "create_llm",
]
