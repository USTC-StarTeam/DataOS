"""Drive one structured LLM call with the runtime's two safety nets.

The skill's autonomous mode failed silently when an orchestrator agent "early-exited";
here the loop owns the call, so an unparseable reply or a flaky endpoint is retried in
code and, only after exhausting retries, raised — never swallowed into a silent stop.

  - parse retries: re-ask ("JSON only") when ``output=`` can't be parsed.
  - network retries: exponential backoff on transient endpoint errors
    (the claude provider has no built-in retry).
"""

from __future__ import annotations

import logging
import json
import time
from typing import Any

from dataos.agent.llm._output import OutputParseError, parse_output

from dataos import prompt as _prompt

log = logging.getLogger(__name__)


def run_structured(
    agent: Any,
    task: str,
    model: Any,
    *,
    parse_retries: int = 2,
    net_retries: int = 3,
    sleep=time.sleep,
) -> Any:
    """Run ``agent.run(task, output=model)`` and return the validated instance.

    Raises the last exception only after retries are exhausted."""
    name = getattr(model, "__name__", str(model))
    prompt = task
    parse_left, net_left = parse_retries, net_retries
    while True:
        try:
            result = agent.run(prompt, output=model)
            out = result.output
            if out is None:
                raise OutputParseError("facade returned output=None")
            return out
        except OutputParseError as e:
            if parse_left <= 0:
                log.error("%s: output unparseable, retries exhausted: %s", name, e)
                raise
            parse_left -= 1
            log.warning("%s: output parse failed (%s); %d parse-retry left, re-asking JSON-only",
                        name, e, parse_left)
            prompt = _prompt.render("structured_json_only", task=task)
        except Exception as e:  # transient endpoint/SDK error: back off and retry
            if net_left <= 0:
                log.error("%s: LLM call failed, net-retries exhausted: %s: %s",
                          name, type(e).__name__, e)
                raise
            net_left -= 1
            backoff = min(2 ** (net_retries - net_left), 30)
            log.warning("%s: transient LLM error %s: %s; %d net-retry left, backoff %ss",
                        name, type(e).__name__, e, net_left, backoff)
            sleep(backoff)


def run_structured_dialog(
    agent: Any,
    task: str,
    model: Any,
    *,
    session_id: str | None = None,
    parse_retries: int = 2,
    net_retries: int = 3,
    sleep=time.sleep,
) -> tuple[Any, str | None]:
    """Run one structured call, preserving a Claude Code dialog when available.

    ``ClaudeCodeAgent.run(output=...)`` parses inside the call, which loses the
    session id on parse failure. For dialog-capable agents we therefore ask for raw
    text, parse it here, and if it is invalid, resume the same session with a
    correction prompt. Non-dialog backends fall back to ``run_structured``.
    """
    if not getattr(agent, "supports_dialog", False):
        return run_structured(
            agent,
            task,
            model,
            parse_retries=parse_retries,
            net_retries=net_retries,
            sleep=sleep,
        ), session_id

    name = getattr(model, "__name__", str(model))
    schema = json.dumps(model.model_json_schema(), ensure_ascii=False)
    base_prompt = _prompt.render("structured_schema", task=task, schema=schema)
    prompt = base_prompt
    sid = session_id
    parse_left, net_left = parse_retries, net_retries

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
        except Exception as e:  # transient endpoint/SDK error
            if net_left <= 0:
                log.error(
                    "%s: dialog LLM call failed, net-retries exhausted: %s: %s",
                    name,
                    type(e).__name__,
                    e,
                )
                raise
            net_left -= 1
            backoff = min(2 ** (net_retries - net_left), 30)
            log.warning(
                "%s: transient dialog LLM error %s: %s; %d net-retry left, backoff %ss",
                name,
                type(e).__name__,
                e,
                net_left,
                backoff,
            )
            sleep(backoff)
