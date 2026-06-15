"""Construct the dataos-agent ``Agent`` for a role — the single place that knows
about endpoints, keys and backend choice.

- ``backend == "claude"`` → Claude Agent SDK backend pointed at an Anthropic-compatible
  endpoint (MiMo ``/anthropic`` by default). Key comes from env, never from config.
- ``backend == "stub"``  → the deterministic ``fake`` provider, constructed directly via
  ``AgentConfig`` (no Agent.fake() factory exists in the package; the direct path needs
  no change to dataos-agent). ``extra["reply"]`` is a fixed JSON string that the
  ``output=`` parser turns back into the role's schema.
"""

from __future__ import annotations

import os
from pathlib import Path

from dataos_agent import Agent, AgentConfig

from .config import AutoRunConfig
from .prompts import PROMPTS


def make_agent(role: str, cfg: AutoRunConfig):
    """Build the agent for ``role`` ("planner" | "reviewer").

    planner/reviewer produce no files (only ``result.output``), so they share a
    throwaway workspace under the project dir."""
    # kept OUTSIDE the project dir — Project.studies lists every subdir of the project
    # dir, so an "_agent" dir there would be mis-counted as a study.
    ws = str(Path(cfg.workspace) / "_agent" / cfg.project_id / role)
    Path(ws).mkdir(parents=True, exist_ok=True)

    if cfg.backend == "stub":
        reply = cfg.stub_replies.get(role, "")
        return Agent(
            AgentConfig(provider="fake", model="stub", workspace=ws, extra={"reply": reply})
        )

    if cfg.backend == "claude":
        key = os.environ.get("MIMO_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "claude backend needs an API key in env: set MIMO_API_KEY "
                "(or ANTHROPIC_API_KEY). The key is never read from config or written to disk."
            )
        return Agent.claude(
            model=cfg.model,
            api_key=key,
            base_url=cfg.base_url,
            instructions=PROMPTS[role],
            workspace=ws,
            max_steps=cfg.max_steps,
        )

    if cfg.backend == "openai":
        # OpenAI-compatible endpoint (DeepSeek/Qwen/local) via base_url — dataos-agent's
        # openai backend uses OpenAIChatCompletionsModel (/chat/completions, not Responses).
        # planner/reviewer only emit structured output, so no tools.
        key = (
            os.environ.get("DPSK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("MIMO_API_KEY")
        )
        if not key:
            raise RuntimeError(
                "openai backend needs an API key in env: set DPSK_API_KEY or OPENAI_API_KEY. "
                "The key is never read from config or written to disk."
            )
        return Agent.openai(
            model=cfg.model,
            api_key=key,
            base_url=cfg.base_url,
            instructions=PROMPTS[role],
            workspace=ws,
            max_steps=cfg.max_steps,
            tools=[],
        )

    raise ValueError(f"unknown backend {cfg.backend!r}; use 'claude', 'openai', or 'stub'")
