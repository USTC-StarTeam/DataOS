"""Prompt registry — every prompt lives as a YAML template here, not hardcoded in code.

Each ``<name>.yaml`` in this folder has a ``template:`` string (Jinja2). Render it with
config / context params:

    from dataos import prompt
    prompt.render("main_agent", budget_hours=2, gpus=1)

Facts (budget, GPUs, the per-experiment context, …) are passed in as render params, so nothing
run-specific is baked into the text. Loading is cached; templates use ``StrictUndefined`` so a
missing variable fails loudly instead of rendering an empty hole.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jinja2
import yaml

_DIR = Path(__file__).resolve().parent
_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


@lru_cache(maxsize=None)
def _template_text(name: str) -> str:
    path = _DIR / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "template" not in data:
        raise ValueError(f"prompt {name!r}: {path} must be a mapping with a 'template' key")
    return str(data["template"])


def render(name: str, /, **params) -> str:
    """Render the prompt template ``name`` with the given params (returns stripped text)."""
    return _ENV.from_string(_template_text(name)).render(**params).strip()


__all__ = ["render"]
