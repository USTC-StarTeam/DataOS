from __future__ import annotations

import os

from dataos.llm.openai_compat import DASHSCOPE_BASE_URL, DEFAULT_MODEL, OpenAICompatLLMClient


def build_operator_llm_client(*, base_url: str | None = None, api_key: str | None = None, model: str | None = None) -> OpenAICompatLLMClient:
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("LLM-backed operators require OPENAI_API_KEY")
    resolved_base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or DASHSCOPE_BASE_URL
    )
    resolved_model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    return OpenAICompatLLMClient(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model=resolved_model,
    )
