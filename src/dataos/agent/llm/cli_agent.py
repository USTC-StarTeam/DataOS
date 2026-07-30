"""The 'cli' agent backend — drive a code-agent CLI (claude / kimi / qwen ...) via dataos_agent.

The brain's agents run on whatever CLI ``provider`` names (dataos_agent's registry);
connection settings are ambient env read by that CLI itself (its own account login or
endpoint env vars) — never by DataOS. Transport self-healing (endpoint drops → backoff
→ resume-reconnect) lives INSIDE dataos_agent, so every caller of this adapter inherits
it for free.

Structured output presents the same ``.run(task, output=Model) -> result.output``
contract the rest of the runtime uses: the schema is rendered into the prompt and the
final text is parsed with :func:`dataos.agent.llm._output.parse_output`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dataos import prompt as _prompt


class AgentError(RuntimeError):
    """A genuine (non-transport) failure from the cli agent backend.

    Raised for a real ``error``/``disconnected`` result (bad auth, reconnect budget
    exhausted, etc.). Never a transient transport fault — those are retried inside
    dataos_agent before a result ever reaches here."""


class CliAgent:
    """The runtime's ``.run(task, *, output=...)`` agent contract, backed by a
    code-agent CLI through dataos_agent (provider-agnostic)."""

    supports_dialog = True

    def __init__(
        self,
        *,
        instructions: str = "",
        workspace: str = ".",
        provider: str = "claude",
        model: str | None = None,
        max_steps: int = 12,
        transcript: str | None = None,
    ) -> None:
        self.instructions = instructions or ""
        self.workspace = str(workspace)
        self.provider = provider
        self.model = model  # None -> the CLI's default model
        self.max_steps = max_steps
        self.transcript = transcript  # provider-neutral JSONL run record (see dataos_agent)
        Path(self.workspace).mkdir(parents=True, exist_ok=True)

    def run(self, task: str, *, output: Any = None, resume: Any = None,
            timeout: float | None = None) -> SimpleNamespace:
        import dataos_agent

        from dataos.agent.llm._output import parse_output

        prompt = task
        if output is not None:
            schema = json.dumps(output.model_json_schema())
            prompt = _prompt.render("task/structured_schema", task=task, schema=schema)

        log = logging.getLogger(__name__)
        # ``timeout`` is the wall-clock budget for the whole call and COVERS dataos_agent's
        # transport reconnects; on expiry dataos_agent raises TimeoutError (not retried).
        try:
            result = asyncio.run(
                dataos_agent.aquery(
                    prompt,
                    provider=self.provider,
                    model=self.model,
                    instructions=self.instructions or None,
                    workspace=self.workspace,
                    max_steps=max(2, self.max_steps),
                    tools=True,
                    permission="auto",
                    timeout=timeout,
                    transcript=self.transcript,
                    extra={"resume": str(resume)} if resume else {},
                )
            )
        except TimeoutError as e:
            raise RuntimeError(f"cli_agent timed out after {timeout}s") from e

        text = result.text
        # max_steps (the CLI's max-turns) is a benign, RESUMABLE boundary for a free-form
        # executor agent (``output is None``): the caller drives it in max-turns-sized
        # chunks and resumes its session until it writes its terminal artifact, so we
        # return normally here (carrying session_id) instead of failing the whole
        # experiment. Structured single-shot calls (planner/reviewer, ``output`` set)
        # genuinely fail when they run out of turns without a complete answer.
        if result.status != "done" and not (result.status == "max_steps" and output is None):
            raise AgentError(
                f"cli_agent error ({result.status}): {result.error or text[:200]}"
            )

        if output is not None:
            # log a short preview so a downstream parse failure is debuggable from the run log
            preview = " ".join(text[:300].split())
            log.info(
                "cli_agent response (%d chars): %s%s", len(text), preview,
                "…" if len(text) > 300 else "",
            )
        parsed = parse_output(text, output) if output is not None else None
        return SimpleNamespace(
            text=text,
            output=parsed,
            plan=[],
            files={},
            steps=[],
            session_id=result.session_id,
        )
