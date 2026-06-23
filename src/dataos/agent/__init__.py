"""DataOS autonomous runtime (agent-version).

A code-driven Agent SDK loop that reimplements the skill's full-auto execution mode:
the loop, fan-out, budget, stop-decision and resume are *owned Python code* (runtime
enforcement); only "write the next plan" and "read the review, set direction" call an
LLM via the dataos-agent Claude SDK backend. The deterministic executor library
(``dataos``, v0.2.5) is unchanged underneath.
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
