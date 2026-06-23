"""Construct the dataos-agent ``Agent`` for a role — the single place that knows
about endpoints, keys and backend choice.

- ``backend == "claude_code"`` → local Claude Code via the CLI's own Claude-account
  (subscription) login — NO API key. Used for the persistent main-agent dialog and
  execution sub-agents. See :mod:`dataos.integrations.llm.claude_code`.
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
from types import SimpleNamespace
from typing import Any

from dataos_agent import Agent, AgentConfig
from dataos_agent._output import parse_output

from .config import AutoRunConfig
from .prompts import PROMPTS


class _StubMainAgent:
    """Tiny structured fake used by tests for the unified main-agent role."""

    supports_dialog = False

    def __init__(self, replies: dict[str, Any]) -> None:
        self.replies = dict(replies)

    def _reply(self, *keys: str) -> str:
        for key in keys:
            if key not in self.replies:
                continue
            value = self.replies[key]
            if isinstance(value, list):
                if len(value) > 1:
                    return str(value.pop(0))
                return str(value[0]) if value else "{}"
            return str(value)
        return "{}"

    def run(self, task: str, *, output=None, resume=None) -> SimpleNamespace:  # noqa: ARG002
        if output is None:
            text = self.replies.get("raw", "{}")
            return SimpleNamespace(text=text, output=None, session_id="stub-main-agent")
        name = getattr(output, "__name__", "")
        if name == "ReviewOut":
            text = self._reply("review", "reviewer", "main_agent_review")
        else:
            text = self._reply("plan", "planner", "main_agent_plan")
        return SimpleNamespace(
            text=text,
            output=parse_output(text, output),
            session_id="stub-main-agent",
        )


def make_agent(role: str, cfg: AutoRunConfig):
    """Build the agent for ``role`` ("main_agent" | legacy "planner"/"reviewer").

    The main-agent produces no files directly, so its workspace lives outside the
    project dir; otherwise Project.studies would mis-count it as a study."""
    # kept OUTSIDE the project dir — Project.studies lists every subdir of the project
    # dir, so an "_agent" dir there would be mis-counted as a study.
    ws = str(Path(cfg.workspace) / "_agent" / cfg.project_id / role)
    Path(ws).mkdir(parents=True, exist_ok=True)

    if cfg.backend == "stub":
        if role == "main_agent":
            return _StubMainAgent(cfg.stub_replies)
        reply = cfg.stub_replies.get(role, "")
        return Agent(
            AgentConfig(provider="fake", model="stub", workspace=ws, extra={"reply": reply})
        )

    if cfg.backend == "claude_code":
        # Local Claude Code with the CLI's own Claude-account (subscription) login — no API key.
        from dataos.integrations.llm.claude_code import ClaudeCodeAgent

        return ClaudeCodeAgent(
            instructions=PROMPTS[role],
            workspace=ws,
            model=cfg.claude_code_model,  # None -> CLI default subscription model
            max_steps=cfg.max_steps,
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
        # The main-agent only emits structured output, so no tools.
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

    raise ValueError(
        f"unknown backend {cfg.backend!r}; use 'claude_code', 'claude', 'openai', or 'stub'"
    )
