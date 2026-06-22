"""A 'claude_code' agent backend — the local Claude Code, logged in with a Claude account.

When the autonomous loop runs locally (mac) and the planner/reviewer brain is Claude Code's
own subscription login (NOT an API key, NOT a third-party endpoint), this drives the local
``claude`` CLI via claude-agent-sdk's ``query()`` with NO ``ANTHROPIC_API_KEY`` injected, so
the SDK uses the CLI's stored login. Structured output reuses ``dataos_agent._output.parse_output``,
so it presents the same ``.run(task, output=Model) -> result.output`` contract the planner /
reviewer already use — no change to the (compiled) dataos-agent package.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _final_text(messages: list) -> str:
    """Reduce a claude-agent-sdk message stream to the final assistant text.

    Mirrors dataos_agent's claude provider: the ResultMessage has ``.result`` and no
    ``.content``; fall back to concatenated assistant text blocks."""
    final = ""
    for m in messages:
        result = getattr(m, "result", None)
        if isinstance(result, str) and result and not hasattr(m, "content"):
            final = result
    if final:
        return final
    parts: list[str] = []
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def _session_id(messages: list) -> str | None:
    """Return the CLI session id emitted by the SDK stream, when available."""
    for m in reversed(messages):
        sid = getattr(m, "session_id", None)
        if isinstance(sid, str) and sid:
            return sid
        data = getattr(m, "data", None)
        if isinstance(data, dict):
            sid = data.get("session_id") or data.get("sessionId")
            if isinstance(sid, str) and sid:
                return sid
    return None


class ClaudeCodeAgent:
    """Same ``.run(task, *, output=...)`` contract as ``dataos_agent.Agent``, backed by the
    local claude CLI's Claude-account (subscription) login — no API key."""

    def __init__(
        self,
        *,
        instructions: str = "",
        workspace: str = ".",
        model: str | None = None,
        max_steps: int = 12,
    ) -> None:
        self.instructions = instructions or ""
        self.workspace = str(workspace)
        self.model = model  # None -> the CLI's default subscription model
        self.max_steps = max_steps
        Path(self.workspace).mkdir(parents=True, exist_ok=True)

    def run(self, task: str, *, output: Any = None, resume: Any = None,
            timeout: float | None = None) -> SimpleNamespace:
        from claude_agent_sdk import ClaudeAgentOptions, query
        from dataos_agent._output import parse_output

        prompt = task
        if output is not None:
            schema = json.dumps(output.model_json_schema())
            prompt = (
                f"{task}\n\nReply with ONLY a JSON object that validates against this JSON "
                f"schema (no prose, no code fence):\n{schema}"
            )

        opts: dict[str, Any] = {
            # the full Claude Code system prompt + our role instructions appended
            "system_prompt": {"type": "preset", "preset": "claude_code", "append": self.instructions},
            "cwd": self.workspace,
            "permission_mode": "bypassPermissions",
            "max_turns": max(2, self.max_steps),
        }
        if self.model:
            opts["model"] = self.model
        if resume:
            opts["resume"] = str(resume)
        # NOTE: no `env` -> no ANTHROPIC_API_KEY -> the CLI uses its own account login.
        options = ClaudeAgentOptions(**opts)

        async def _collect() -> list:
            collected: list = []
            async for message in query(prompt=prompt, options=options):
                collected.append(message)
            return collected

        async def _collect_with_timeout() -> list:
            return await asyncio.wait_for(_collect(), timeout)

        try:
            messages = asyncio.run(_collect_with_timeout() if timeout else _collect())
        except (asyncio.TimeoutError, TimeoutError) as e:
            raise RuntimeError(f"claude_code timed out after {timeout}s") from e
        for m in messages:
            if getattr(m, "is_error", False):
                raise RuntimeError(f"claude_code error: {getattr(m, 'result', 'unknown')}")

        text = _final_text(messages)
        if output is not None:
            # log a short preview so a downstream parse failure is debuggable from the run log
            preview = " ".join(text[:300].split())
            logging.getLogger(__name__).info(
                "claude_code response (%d chars): %s%s", len(text), preview,
                "…" if len(text) > 300 else "",
            )
        parsed = parse_output(text, output) if output is not None else None
        return SimpleNamespace(
            text=text,
            output=parsed,
            plan=[],
            files={},
            steps=[],
            session_id=_session_id(messages),
        )
