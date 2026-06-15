"""LLM judgment point #1 — write the next round's plan.

Deterministic everywhere except the single ``agent.run(..., output=PlanOut)`` call:
the loop builds the prompt from the prior review + digest + operator catalog, the model
returns a typed PlanOut, and the loop (not the model) decides scheduling and stopping.
"""

from __future__ import annotations

import json

from ._llm import run_structured
from .schemas import PlanOut


def _summarize_catalog(catalog: list[dict] | None, limit: int = 40) -> list[dict]:
    out = []
    for op in (catalog or [])[:limit]:
        out.append(
            {
                "name": op.get("name"),
                "kind": op.get("kind"),
                "params": [p.get("name") for p in (op.get("params") or [])],
            }
        )
    return out


def _build_prompt(*, instruct, goal, prev_review, prev_digest, eval_cfg, round_idx, catalog) -> str:
    parts = [
        f"Round index: {round_idx} (0-based; this is round number {round_idx + 1}).",
        f"User goal: {goal or instruct or '(unspecified)'}",
        f"User instruction: {instruct or '(none)'}",
    ]
    if prev_review is not None:
        parts.append(
            "Previous round review:\n"
            + json.dumps(
                {"conclusion": prev_review.conclusion, "direction": prev_review.direction},
                ensure_ascii=False,
            )
        )
    if prev_digest is not None:
        slim = {
            k: prev_digest.get(k)
            for k in ("ranking", "best", "varied_keys", "metric_winners", "caution")
            if k in prev_digest
        }
        parts.append("Previous round digest:\n" + json.dumps(slim, ensure_ascii=False))
    else:
        parts.append("This is the first round — no prior results.")
    parts.append(
        "The evaluation protocol is FROZEN (injected by the runtime); do not set `eval`. "
        "Frozen eval (for context only):\n" + json.dumps(eval_cfg or {}, ensure_ascii=False)
    )
    parts.append(
        "Training data is prepared once and shared; leave `data` empty. "
        "Available data operators (name/kind/params):\n"
        + json.dumps(_summarize_catalog(catalog), ensure_ascii=False)
    )
    parts.append("Now produce the PlanOut JSON for this round.")
    return "\n\n".join(parts)


def next_plan(
    *,
    agent,
    instruct: str = "",
    goal: str = "",
    prev_review=None,
    prev_digest: dict | None = None,
    eval_cfg: dict | None = None,
    round_idx: int = 0,
    catalog: list[dict] | None = None,
) -> PlanOut:
    """Ask the planner for this round's experiments (typed PlanOut)."""
    task = _build_prompt(
        instruct=instruct,
        goal=goal,
        prev_review=prev_review,
        prev_digest=prev_digest,
        eval_cfg=eval_cfg,
        round_idx=round_idx,
        catalog=catalog,
    )
    return run_structured(agent, task, PlanOut)
