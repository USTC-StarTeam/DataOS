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

from pathlib import Path
from typing import TYPE_CHECKING

from dataos_agent import context_filename

from dataos.agent.llm.cli_agent import CliAgent

if TYPE_CHECKING:
    from dataos.agent.config import AutoRunConfig

# The roles' cwd and config root live OUTSIDE the repo. A cwd inside it makes every CLI
# auto-discover <repo>/AGENTS.md — which is the guide for agents DEVELOPING DataOS
# ("Operating guide for AI coding agents working in this repository"), not for the roles
# RUNNING it. Keyed per role so the three share no context file, session or memory.
_AGENT_ROOT = Path.home() / ".cache" / "dataos-agent"


def role_cwd(project_id: str, role: str) -> str:
    """The working directory for one role of one run.

    Deterministic, so it is the same directory on a later call and after a process restart. It is
    the ONLY isolation this layer does: the CLI's own config root stays shared with the host,
    because that root holds an OAuth credential with a single-use refresh token which cannot be
    replicated across concurrent runs (see ``dataos_agent._home`` for what that cost us)."""
    cwd = _AGENT_ROOT / project_id / role / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    return str(cwd)


def build_agent(
    cfg: "AutoRunConfig",
    *,
    role: str,
    instructions: str,
    max_steps: int | None = None,
    transcript: str | None = None,
) -> CliAgent:
    """Build the ``CliAgent`` for a role.

    ``provider``/``model`` always come from ``cfg.agent_provider`` /
    ``cfg.agent_model`` (``None`` model -> the CLI's default model). ``max_steps``
    defaults to ``cfg.max_steps``; callers that need a bigger turn budget
    (execution experimenters) pass their own. ``transcript`` is the JSONL
    run-record path (tee'd neutral events) when activity should be persisted.

    ``instructions`` is the role's high-level prompt. It is written into the role's cwd
    as the CLI's own global context file (``AGENTS.md`` / ``CLAUDE.md`` / ``QWEN.md``), so
    it rides that CLI's native channel — re-injected on every resume, never part of the
    compactable conversation — instead of being prepended to each turn's prompt. Rewritten
    unconditionally: the prompt source may have changed since the last run.
    """
    cwd = role_cwd(cfg.project_id, role)
    ctx_name = context_filename(cfg.agent_provider)
    if instructions and ctx_name:
        (Path(cwd) / ctx_name).write_text(instructions, encoding="utf-8")
        instructions = ""  # delivered as the context file; don't also send it per turn
    return CliAgent(
        instructions=instructions,
        workspace=cwd,
        provider=cfg.agent_provider,
        model=cfg.agent_model,
        max_steps=cfg.max_steps if max_steps is None else max_steps,
        transcript=transcript,
        effort=cfg.agent_effort,
        idle_timeout=cfg.agent_idle_timeout_s or None,  # 0/None disables the watchdog
    )
