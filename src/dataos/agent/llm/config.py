"""Code-agent CLI connection config + the single constructor for a ``CliAgent``.

Connection settings are ambient env, read by the provider CLI itself — never by
DataOS and never from :class:`AutoRunConfig`:

  - the CLI's own account login (no API key), or
  - its endpoint env vars (e.g. ``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_AUTH_TOKEN``).

So the only settings this layer owns are the *provider*, the *model* and the
per-role *step budget*, taken from the run's :class:`AutoRunConfig` here. Routing
every role through :func:`build_agent` guarantees the leader and the execution
experimenters build the same backend the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataos.agent.llm.cli_agent import CliAgent

if TYPE_CHECKING:
    from dataos.agent.config import AutoRunConfig


def build_agent(
    cfg: "AutoRunConfig",
    *,
    instructions: str,
    workspace: str,
    max_steps: int | None = None,
) -> CliAgent:
    """Build the ``CliAgent`` for a role.

    ``provider``/``model`` always come from ``cfg.agent_provider`` /
    ``cfg.agent_model`` (``None`` model -> the CLI's default model). ``max_steps``
    defaults to ``cfg.max_steps``; callers that need a bigger turn budget
    (execution experimenters) pass their own.
    """
    return CliAgent(
        instructions=instructions,
        workspace=workspace,
        provider=cfg.agent_provider,
        model=cfg.agent_model,
        max_steps=cfg.max_steps if max_steps is None else max_steps,
    )
