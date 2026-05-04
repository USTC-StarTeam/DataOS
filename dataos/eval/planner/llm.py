"""Default LLM-client factory for the eval planner.

Reuses ``dataos.llm.OpenAICompatLLMClient`` and the same env-var conventions as
``dataos.cli._build_llm_client_from_env`` so the planner shares one source of
truth with the rest of DataOS (CLI, runtime agent, research orchestrator).

Recognized env vars (legacy aliases supported):

* ``DASHSCOPE_API_KEY`` / ``STARTEAM_API_KEY``     — required
* ``DASHSCOPE_BASE_URL`` / ``STARTEAM_BASE_URL``   — defaults to DashScope
* ``DASHSCOPE_MODEL`` / ``STARTEAM_MODEL``         — defaults to ``deepseek-v4-flash``
"""

from __future__ import annotations

import os

from dataos.llm import (
    DASHSCOPE_BASE_URL,
    DEFAULT_MODEL,
    LLMClient,
    OpenAICompatLLMClient,
)


def default_llm_client() -> LLMClient:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("STARTEAM_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing DASHSCOPE_API_KEY (or legacy STARTEAM_API_KEY) for the eval planner LLM"
        )
    base_url = (
        os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("STARTEAM_BASE_URL")
        or DASHSCOPE_BASE_URL
    )
    model = (
        os.environ.get("DASHSCOPE_MODEL")
        or os.environ.get("STARTEAM_MODEL")
        or DEFAULT_MODEL
    )
    return OpenAICompatLLMClient(api_key=api_key, base_url=base_url, model=model)
