"""A 'claude_code' agent backend — the local Claude Code, logged in with a Claude account.

When the autonomous loop runs locally (mac) and the main-agent brain is Claude Code's
own subscription login (NOT an API key, NOT a third-party endpoint), this drives the local
``claude`` CLI via claude-agent-sdk's ``query()`` with NO ``ANTHROPIC_API_KEY`` injected, so
the SDK uses the CLI's stored login. Structured output uses
:func:`dataos.agent.llm._output.parse_output`, so it presents the same
``.run(task, output=Model) -> result.output`` contract the rest of the runtime uses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dataos import prompt as _prompt
from dataos.agent.llm.net_retry import (
    MAX_NET_RETRIES,
    ClaudeCodeError,
    backoff_seconds,
    is_recoverable,
)


def _final_text(messages: list) -> str:
    """Reduce a claude-agent-sdk message stream to the final assistant text.

    The final ResultMessage has ``.result`` and no ``.content``; fall back to
    concatenated assistant text blocks when no such result is present."""
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
    """The runtime's ``.run(task, *, output=...)`` agent contract, backed by the
    local claude CLI's Claude-account (subscription) login — no API key."""

    supports_dialog = True

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

        from dataos.agent.llm._output import parse_output

        prompt = task
        if output is not None:
            schema = json.dumps(output.model_json_schema())
            prompt = _prompt.render("structured_schema", task=task, schema=schema)

        log = logging.getLogger(__name__)
        # Transport-level self-healing (shared by main-agent + sub-agents, since both go
        # through here): the Anthropic-compatible endpoints drop connections mid-response
        # and the SDK has no auto-reconnect. On a recognised transient fault we back off and
        # RESUME the same session so no work is lost. Bounded by the caller's ``timeout``
        # budget when set, and by MAX_NET_RETRIES otherwise (the main-agent has no timeout).
        deadline = time.monotonic() + timeout if timeout else None
        resume_sid = str(resume) if resume else None
        collected: list = []
        attempt = 0

        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                raise RuntimeError(f"claude_code timed out after {timeout}s")

            opts: dict[str, Any] = {
                # the full Claude Code system prompt + our role instructions appended
                "system_prompt": {"type": "preset", "preset": "claude_code", "append": self.instructions},
                "cwd": self.workspace,
                "permission_mode": "bypassPermissions",
                "max_turns": max(2, self.max_steps),
            }
            if self.model:
                opts["model"] = self.model
            if resume_sid:
                opts["resume"] = resume_sid
            # NOTE: no `env` -> no ANTHROPIC_API_KEY -> the CLI uses its own account login.
            options = ClaudeAgentOptions(**opts)

            collected = []

            async def _collect() -> None:
                try:
                    async for message in query(prompt=prompt, options=options):
                        collected.append(message)
                except Exception:
                    # On error_max_turns the CLI emits the is_error result (which carries
                    # the session_id) and THEN exits non-zero, so query() raises right after
                    # yielding it. Keep what we collected so the caller can read that result
                    # and resume the session; only propagate if no terminating result arrived.
                    if not any(getattr(m, "is_error", False) for m in collected):
                        raise

            try:
                if remaining is not None:
                    asyncio.run(asyncio.wait_for(_collect(), remaining))
                else:
                    asyncio.run(_collect())
                break  # stream finished (or ended on a preserved is_error result)
            except (asyncio.TimeoutError, TimeoutError) as e:
                # the wall-clock budget is spent — not a transport fault; don't retry.
                raise RuntimeError(f"claude_code timed out after {timeout}s") from e
            except Exception as e:  # noqa: BLE001 — classified by is_recoverable (whitelist)
                if not is_recoverable(e):
                    raise
                attempt += 1
                # resume the session learned before the drop (falls back to the caller's).
                resume_sid = _session_id(collected) or resume_sid
                if deadline is None and attempt > MAX_NET_RETRIES:
                    raise RuntimeError(
                        f"claude_code: transport error persisted after {MAX_NET_RETRIES} "
                        f"reconnects: {type(e).__name__}: {e}"
                    ) from e
                wait = backoff_seconds(attempt)
                if deadline is not None:
                    wait = min(wait, max(0.0, deadline - time.monotonic()))
                log.warning(
                    "claude_code: transport error %s: %s; reconnect #%d (resume=%s) in %.0fs",
                    type(e).__name__, e, attempt, resume_sid, wait,
                )
                time.sleep(wait)

        messages = collected
        for m in messages:
            if getattr(m, "is_error", False):
                errs = getattr(m, "errors", None) or []
                subtype = str(getattr(m, "subtype", "") or "")
                detail = str(
                    getattr(m, "result", None) or "; ".join(map(str, errs)) or subtype or "unknown"
                )
                # max_turns is a benign, RESUMABLE boundary for a free-form executor agent
                # (``output is None``): the caller drives it in max_turns-sized chunks and
                # resumes its session until it writes its terminal artifact, so we must return
                # normally here (carrying session_id) instead of failing the whole experiment.
                # Structured single-shot calls (planner/reviewer, ``output`` set) genuinely fail
                # when they run out of turns without a complete answer, so those still raise.
                hit_max_turns = "max_turns" in subtype or "maximum number of turns" in detail
                if output is None and hit_max_turns:
                    break
                raise ClaudeCodeError(f"claude_code error: {detail}")

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
