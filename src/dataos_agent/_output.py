"""Portable structured output: pull a JSON object out of the agent's final text
(or a file it wrote) and validate it against a pydantic schema. Backend-neutral —
we never rely on a provider's native structured-output feature.

The agent is instructed to emit its result as JSON; this tolerates the common
shapes: raw JSON, a ```json fenced block, or JSON embedded in prose/thinking."""

from __future__ import annotations

import json
import os
import re

from pydantic import BaseModel, ValidationError

from .errors import OutputParseError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _candidates(text: str):
    """Yield JSON-string candidates from ``text``, most-specific first."""
    stripped = text.strip()
    if stripped:
        yield stripped                       # the whole thing is JSON
    for m in _FENCE.findall(text):           # ```json ... ``` blocks
        if m.strip():
            yield m.strip()
    for chunk in _balanced(text):            # embedded {...} / [...]
        yield chunk


def _balanced(s: str):
    """Yield top-level balanced {...} or [...] substrings (ignoring braces in strings)."""
    opens = {"{": "}", "[": "]"}
    i = 0
    while i < len(s):
        if s[i] in opens:
            close = opens[s[i]]
            depth = 0
            in_str = False
            esc = False
            for j in range(i, len(s)):
                c = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == s[i]:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        yield s[i : j + 1]
                        i = j
                        break
        i += 1


def parse_output(text_or_path: str, schema: type[BaseModel]) -> BaseModel:
    """Return a validated ``schema`` instance parsed from the agent's output.

    ``text_or_path``: the agent's final text, or a path to a file it wrote.
    Raises ``OutputParseError`` if no candidate validates.
    """
    text = text_or_path
    if isinstance(text_or_path, str) and os.path.sep in text_or_path and os.path.isfile(text_or_path):
        text = open(text_or_path, encoding="utf-8").read()

    last: Exception | None = None
    seen: set[str] = set()
    for cand in _candidates(text):
        if cand in seen:
            continue
        seen.add(cand)
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            last = e
            continue
        try:
            return schema.model_validate(obj)
        except ValidationError as e:
            last = e
            continue
    raise OutputParseError(
        f"No JSON matching {schema.__name__} found in output "
        f"({len(text)} chars). Last error: {last}"
    )
