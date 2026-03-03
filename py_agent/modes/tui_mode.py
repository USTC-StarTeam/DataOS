"""Interactive TUI mode."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Callable

from py_agent.providers.base import Provider, ProviderError
from py_agent.runtime import run_agent_loop
from py_agent.session import SessionStore
from py_agent.skills import apply_skill_invocations, build_skills_system_prompt, discover_skills
from py_agent.tools import Tool, create_local_tools

InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]


def format_event_lines(events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "tool_call":
            lines.append(
                f"[tool_call] {event.get('tool')} id={event.get('tool_call_id')} args={event.get('arguments')}"
            )
        elif event_type == "tool_result":
            lines.append(
                f"[tool_result] {event.get('tool')} id={event.get('tool_call_id')} content={event.get('content')}"
            )
    return lines


def run_tui_mode(
    provider: Provider,
    model: str,
    cwd: str,
    session_store: SessionStore | None = None,
    initial_messages: list[dict[str, Any]] | None = None,
    extra_tools: dict[str, Tool] | None = None,
    input_func: InputFunc | None = None,
    output_func: OutputFunc = print,
) -> int:
    if input_func is None:
        input_func = _build_default_input()

    tools = create_local_tools(cwd=cwd)
    if extra_tools:
        tools.update(extra_tools)

    skills = discover_skills(cwd)
    system_prompt = build_skills_system_prompt(skills)
    conversation_messages: list[dict[str, Any]] = list(initial_messages or [])

    output_func("TUI mode started. Type /exit or /quit to leave.")
    if conversation_messages:
        output_func(f"Loaded {len(conversation_messages)} prior messages.")

    while True:
        try:
            raw = input_func("tui> ")
        except (KeyboardInterrupt, EOFError):
            output_func("")
            return 0

        prompt = raw.strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0

        resolved_prompt = apply_skill_invocations(prompt, skills)
        turn_system_prompt = system_prompt if not any(m.get("role") == "system" for m in conversation_messages) else None
        try:
            result = run_agent_loop(
                provider=provider,
                model=model,
                prompt=resolved_prompt,
                tools=tools,
                initial_messages=conversation_messages,
                system_prompt=turn_system_prompt,
            )
        except ProviderError as exc:
            output_func(f"provider error: {exc}")
            continue

        output_func(result.text)
        for line in format_event_lines(result.events):
            output_func(line)

        conversation_messages.extend(result.turn_messages)
        if session_store:
            session_store.append_turn(messages=result.turn_messages, events=result.events)


def _default_input(prompt: str) -> str:
    # Avoid Python readline/libedit edge cases with CJK input editing.
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\n")


def _build_default_input() -> InputFunc:
    # Prefer prompt_toolkit for robust CJK/IME line editing behavior.
    try:
        from prompt_toolkit import PromptSession
    except Exception:
        _prepare_terminal_for_utf8_input()
        return _default_input

    session = PromptSession()

    def _pt_input(prompt: str) -> str:
        return session.prompt(prompt)

    return _pt_input


def _prepare_terminal_for_utf8_input() -> None:
    # Best effort: on macOS/Linux, ensure terminal erases UTF-8 chars correctly.
    if os.name == "nt":
        return
    if not sys.stdin.isatty():
        return
    try:
        subprocess.run(
            ["stty", "iutf8"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
