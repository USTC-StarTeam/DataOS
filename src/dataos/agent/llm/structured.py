"""Drive one structured LLM call with a parse-correction safety net.

The skill's autonomous mode failed silently when an orchestrator agent "early-exited";
here the loop owns the call, so an unparseable reply is re-asked in code and, only after
exhausting retries, raised — never swallowed into a silent stop.

Transport faults (dropped connections, flaky endpoints) are handled ONE LAYER DOWN, inside
dataos_agent's self-healing stream (reached via
:meth:`dataos.agent.llm.cli_agent.CliAgent.run`), so this module owns only the *semantic*
retry: re-ask "JSON only" when ``output=`` cannot be parsed. A transport error only
reaches here if the backend could not recover it, and it is intentionally raised rather
than retried again.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from dataos import prompt as _prompt
from dataos.agent.llm._output import OutputParseError, parse_output

log = logging.getLogger(__name__)


def run_structured(
    agent: Any,
    task: str,
    model: Any,
    *,
    parse_retries: int = 2,
) -> Any:
    """Run ``agent.run(task, output=model)`` and return the validated instance.

    Re-asks JSON-only on a parse failure; raises the last error only after
    ``parse_retries`` are exhausted."""
    name = getattr(model, "__name__", str(model))
    prompt = task
    parse_left = parse_retries
    while True:
        try:
            result = agent.run(prompt, output=model)
            out = result.output
            if out is None:
                raise OutputParseError("backend returned output=None")
            return out
        except OutputParseError as e:
            if parse_left <= 0:
                log.error("%s: output unparseable, retries exhausted: %s", name, e)
                raise
            parse_left -= 1
            log.warning("%s: output parse failed (%s); %d parse-retry left, re-asking JSON-only",
                        name, e, parse_left)
            prompt = _prompt.render("structured_json_only", task=task)


def run_structured_dialog(
    agent: Any,
    task: str,
    model: Any,
    *,
    session_id: str | None = None,
    parse_retries: int = 2,
) -> tuple[Any, str | None]:
    """Run one structured call, preserving the agent CLI's dialog when available.

    ``CliAgent.run(output=...)`` parses inside the call, which loses the
    session id on parse failure. For dialog-capable agents we therefore ask for raw
    text, parse it here, and if it is invalid, resume the same session with a
    correction prompt. Non-dialog backends fall back to ``run_structured``.
    """
    if not getattr(agent, "supports_dialog", False):
        return run_structured(agent, task, model, parse_retries=parse_retries), session_id

    name = getattr(model, "__name__", str(model))
    schema = json.dumps(model.model_json_schema(), ensure_ascii=False)
    base_prompt = _prompt.render("structured_schema", task=task, schema=schema)
    prompt = base_prompt
    sid = session_id
    parse_left = parse_retries

    while True:
        try:
            result = agent.run(prompt, resume=sid)
            sid = getattr(result, "session_id", None) or sid
            text = getattr(result, "text", "") or ""
            preview = " ".join(text[:300].split())
            log.info(
                "%s dialog response (%d chars, session=%s): %s%s",
                name,
                len(text),
                sid,
                preview,
                "…" if len(text) > 300 else "",
            )
            return parse_output(text, model), sid
        except OutputParseError as e:
            if parse_left <= 0:
                log.error("%s: dialog output unparseable, retries exhausted: %s", name, e)
                raise
            parse_left -= 1
            log.warning(
                "%s: dialog output parse failed (%s); %d parse-retry left",
                name,
                e,
                parse_left,
            )
            prompt = _prompt.render("structured_correction", error=e, schema=schema)
