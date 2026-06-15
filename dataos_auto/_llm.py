"""Drive one structured LLM call with the runtime's two safety nets.

The skill's autonomous mode failed silently when an orchestrator agent "early-exited";
here the loop owns the call, so an unparseable reply or a flaky endpoint is retried in
code and, only after exhausting retries, raised — never swallowed into a silent stop.

  - parse retries: re-ask ("JSON only") when ``output=`` can't be parsed.
  - network retries: exponential backoff on transient endpoint errors
    (the claude provider has no built-in retry).
"""

from __future__ import annotations

import time
from typing import Any

from dataos_agent.errors import OutputParseError


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
    prompt = task
    parse_left, net_left = parse_retries, net_retries
    while True:
        try:
            result = agent.run(prompt, output=model)
            out = result.output
            if out is None:
                raise OutputParseError("facade returned output=None")
            return out
        except OutputParseError:
            if parse_left <= 0:
                raise
            parse_left -= 1
            prompt = (
                task
                + "\n\nIMPORTANT: your previous reply was not valid JSON. Reply with ONLY "
                "the JSON object — no prose, no markdown fence."
            )
        except Exception:  # transient endpoint/SDK error: back off and retry
            if net_left <= 0:
                raise
            net_left -= 1
            sleep(min(2 ** (net_retries - net_left), 30))
