"""Leader review prompt for one completed tree node."""

from __future__ import annotations

import json

from dataos import prompt as _prompt
from dataos.agent.llm.structured import run_structured_dialog

from .schemas import ReviewOut


def _build_prompt(*, context: dict) -> str:
    return _prompt.render(
        "task/review",
        context_json=json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True),
    )


def review_result(
    *,
    agent,
    context: dict,
    session_id: str | None = None,
) -> tuple[ReviewOut, str | None]:
    """Ask the leader dialog to interpret one completed node."""

    return run_structured_dialog(agent, _build_prompt(context=context), ReviewOut, session_id=session_id)


def conclude(*, agent, digest: dict, goal: str = "", round_idx: int = 0) -> ReviewOut:
    """Compatibility wrapper for older callers that expected a stateless reviewer."""

    context = {"goal": goal, "round_idx": round_idx, "study_digest": digest}
    review, _ = review_result(agent=agent, context=context)
    return review
