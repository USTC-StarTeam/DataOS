"""Print mode handling for one-shot prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass

from py_agent.providers.base import Provider
from py_agent.runtime import run_agent_loop
from py_agent.tools import create_local_tools


@dataclass(frozen=True)
class PrintModeResult:
    model: str
    output: str


def run_print_mode(provider: Provider, prompt: str, model: str, mode: str, cwd: str = ".") -> PrintModeResult:
    tools = create_local_tools(cwd=cwd)
    completion = run_agent_loop(provider=provider, model=model, prompt=prompt, tools=tools)
    if mode == "text":
        return PrintModeResult(model=completion.model, output=completion.text)

    event = {
        "type": "final",
        "model": completion.model,
        "content": completion.text,
    }
    return PrintModeResult(model=completion.model, output=json.dumps(event, ensure_ascii=False))
