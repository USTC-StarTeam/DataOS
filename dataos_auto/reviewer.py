"""LLM judgment point #2 — read the study digest, write conclusion + direction.

The reviewer does NOT decide whether to stop (that is the runtime budget's job). It
only produces a grounded conclusion and a steer for the next round's planner.
"""

from __future__ import annotations

import json

from ._llm import run_structured
from .schemas import ReviewOut


def _build_prompt(*, digest: dict, goal: str, round_idx: int) -> str:
    slim = {
        k: digest.get(k)
        for k in ("ranking", "best", "varied_keys", "metric_winners", "caution", "n_experiments")
        if k in digest
    }
    return "\n\n".join(
        [
            f"Round index: {round_idx}.",
            f"User goal: {goal or '(unspecified)'}",
            "Study digest:\n" + json.dumps(slim, ensure_ascii=False),
            "Now produce the ReviewOut JSON (conclusion + direction). "
            "If `caution` is set, say the ranking is unreliable and recommend a re-run.",
        ]
    )


def conclude(*, agent, digest: dict, goal: str = "", round_idx: int = 0) -> ReviewOut:
    """Ask the reviewer for a conclusion + next-round direction (typed ReviewOut)."""
    task = _build_prompt(digest=digest, goal=goal, round_idx=round_idx)
    return run_structured(agent, task, ReviewOut)
