"""Claude Code connection config + the single constructor for a ``ClaudeCodeAgent``.

Connection settings are ambient env, read by the ``claude`` CLI itself — never by
DataOS and never from :class:`AutoRunConfig`:

  - a Claude-account subscription login (no API key), or
  - ``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_AUTH_TOKEN`` for a compatible endpoint.

So the only cc settings this layer owns are the *model* and the per-role *step
budget*, both taken from the run's :class:`AutoRunConfig` here. Routing every role
through :func:`build_agent` guarantees the main-agent and the execution sub-agents
build the same backend the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataos.agent.llm.claude_code import ClaudeCodeAgent

if TYPE_CHECKING:
    from dataos.agent.config import AutoRunConfig


def build_agent(
    cfg: "AutoRunConfig",
    *,
    instructions: str,
    workspace: str,
    max_steps: int | None = None,
) -> ClaudeCodeAgent:
    """Build the ``ClaudeCodeAgent`` for a role.

    ``model`` always comes from ``cfg.claude_code_model`` (``None`` -> the CLI's
    default subscription model). ``max_steps`` defaults to ``cfg.max_steps``; callers
    that need a bigger turn budget (execution sub-agents) pass their own.
    """
    return ClaudeCodeAgent(
        instructions=instructions,
        workspace=workspace,
        model=cfg.claude_code_model,
        max_steps=cfg.max_steps if max_steps is None else max_steps,
    )
