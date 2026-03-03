"""Print mode handling for one-shot prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from py_agent.providers.base import Provider
from py_agent.runtime import run_agent_loop
from py_agent.skills import apply_skill_invocations, build_skills_system_prompt, discover_skills
from py_agent.tools import Tool, create_local_tools


@dataclass(frozen=True)
class PrintModeResult:
    model: str
    output: str
    turn_messages: list[dict[str, Any]]
    events: list[dict[str, Any]]


def run_print_mode(
    provider: Provider,
    prompt: str,
    model: str,
    mode: str,
    cwd: str = ".",
    initial_messages: list[dict[str, Any]] | None = None,
    extra_tools: dict[str, Tool] | None = None,
) -> PrintModeResult:
    tools = create_local_tools(cwd=cwd)
    if extra_tools:
        tools.update(extra_tools)
    skills = discover_skills(cwd)
    resolved_prompt = apply_skill_invocations(prompt, skills)
    system_prompt = build_skills_system_prompt(skills)
    completion = run_agent_loop(
        provider=provider,
        model=model,
        prompt=resolved_prompt,
        tools=tools,
        initial_messages=initial_messages,
        system_prompt=system_prompt or None,
    )
    if mode == "text":
        return PrintModeResult(
            model=completion.model,
            output=completion.text,
            turn_messages=completion.turn_messages,
            events=completion.events,
        )

    event = {
        "type": "final",
        "model": completion.model,
        "content": completion.text,
    }
    return PrintModeResult(
        model=completion.model,
        output=json.dumps(event, ensure_ascii=False),
        turn_messages=completion.turn_messages,
        events=completion.events,
    )
