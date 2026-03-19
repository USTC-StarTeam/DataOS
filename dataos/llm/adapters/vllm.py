"""vLLM adapter backed by the OpenAI-compatible DataOS LLM interface."""

from __future__ import annotations

import os

from dataos.llm.adapters.openai import OpenAICompatibleLLM

BASE_URL_ENV = "VLLM_BASE_URL"
API_KEY_ENV = "VLLM_API_KEY"


class VLLMOpenAIAdapter(OpenAICompatibleLLM):
    """vLLM exposes an OpenAI-compatible API, so the adapter reuses that contract."""

    provider_name = "vllm"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        resolved_base_url = base_url or os.getenv(BASE_URL_ENV, "")
        resolved_api_key = api_key if api_key is not None else os.getenv(API_KEY_ENV, "")
        super().__init__(base_url=resolved_base_url or None, api_key=resolved_api_key, **kwargs)


__all__ = ["API_KEY_ENV", "BASE_URL_ENV", "VLLMOpenAIAdapter"]
