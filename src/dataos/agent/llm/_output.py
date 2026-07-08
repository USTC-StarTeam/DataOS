"""Parse a model's final text into a Pydantic schema for ``output=``.

Self-contained (no external agent package): extract a JSON object/array from free-form
model text and validate it against a Pydantic model. Owned by DataOS so the whole
structured-output path lives in one repo.
"""
from __future__ import annotations

import re
from typing import Any, Iterator


class OutputParseError(Exception):
    """Raised when a model's text cannot be parsed into the requested schema."""


def _balanced(s: str, i: int) -> str | None:
    """Return the balanced bracket fragment starting at ``s[i]``, ignoring brackets that
    appear inside JSON strings. ``None`` if it never balances."""
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
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return s[i : j + 1]
    return None


def _json_candidates(s: str) -> Iterator[str]:
    """Yield each balanced ``{...}``/``[...]`` fragment in order, so a real JSON object
    after some prose (which may itself contain stray braces) can still be found."""
    for i, ch in enumerate(s):
        if ch in "{[":
            frag = _balanced(s, i)
            if frag is not None:
                yield frag


def parse_output(text: str, model: Any) -> Any:
    """Extract a JSON object/array from ``text`` and validate it against ``model``."""
    s = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", s, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else (list(_json_candidates(s)) or [s])
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            return model.model_validate_json(candidate)
        except Exception as exc:  # try the next balanced fragment
            last_exc = exc
    raise OutputParseError(
        f"Could not parse model output into {getattr(model, '__name__', model)}: {last_exc}"
    ) from last_exc
