"""DataOS autonomous runtime.

The main loop is a code-owned state machine around a persistent leader dialog:
Claude Code proposes one next experiment plan and reviews one completed node; Python
validates JSON, updates the tree, injects the parent model, launches execution, and
persists reports/reviews.
"""

from __future__ import annotations

__all__ = ["run_auto", "AutoRunConfig", "Budget", "Criterion"]


def __getattr__(name: str):
    # Lazy so importing the package never eagerly pulls the LLM/SDK stack.
    if name == "run_auto":
        from .loop import run_auto

        return run_auto
    if name in ("AutoRunConfig", "Budget", "Criterion"):
        from . import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
