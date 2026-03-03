"""Print mode handling for one-shot prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass

from py_agent.providers.base import Provider


@dataclass(frozen=True)
class PrintModeResult:
    model: str
    output: str


def run_print_mode(provider: Provider, prompt: str, model: str, mode: str) -> PrintModeResult:
    completion = provider.complete(prompt=prompt, model=model)
    if mode == "text":
        return PrintModeResult(model=completion.model, output=completion.text)

    event = {
        "type": "final",
        "model": completion.model,
        "content": completion.text,
    }
    return PrintModeResult(model=completion.model, output=json.dumps(event, ensure_ascii=False))
