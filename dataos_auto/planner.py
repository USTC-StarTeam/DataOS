"""LLM judgment point #1 — propose one tree EDGE per assigned slot.

The skeleton owns the tree and selects this round's 3 slots (2 exploit + 1 explore); the
planner only fills each slot's recipe (data mixture + SFT hyperparameters). Deterministic
everywhere except the single ``agent.run(..., output=PlanOut)`` call.
"""

from __future__ import annotations

import json

from ._llm import run_structured
from .schemas import PlanOut


def _summarize_catalog(catalog: list[dict] | None, limit: int = 40) -> list[dict]:
    out = []
    for op in (catalog or [])[:limit]:
        out.append({
            "name": op.get("name"),
            "kind": op.get("kind"),
            "params": [p.get("name") for p in (op.get("params") or [])],
        })
    return out


def _build_prompt(*, instruct, goal, prev_review, targets, explored, eval_cfg, round_idx, catalog) -> str:
    parts = [
        f"Round index: {round_idx} (0-based).",
        f"User goal: {goal or instruct or '(unspecified)'}",
    ]
    if prev_review is not None:
        parts.append(
            "Previous round review:\n"
            + json.dumps({"conclusion": prev_review.conclusion, "direction": prev_review.direction},
                         ensure_ascii=False)
        )
    # the slots the skeleton picked this round — fill ONE edge per slot, in order
    slot_lines = []
    for i, t in enumerate(targets):
        if t["role"] == "exploit":
            slot_lines.append(
                f"  slot {i} = EXPLOIT: refine this parent (score={t['parent_score']}, depth={t['parent_depth']}). "
                f"Make a SMALL change to its recipe to push the score up.\n"
                f"    parent recipe: {json.dumps(t['parent_recipe'], ensure_ascii=False)}"
            )
        else:
            slot_lines.append(
                f"  slot {i} = EXPLORE: propose a FRESH recipe from the base, deliberately "
                f"DIFFERENT from everything tried, to open new ground."
            )
    parts.append("Slots to fill this round (one edge each, in this order):\n" + "\n".join(slot_lines))
    if explored:
        parts.append("Recipes already tried (avoid repeating, especially for EXPLORE):\n"
                     + json.dumps(explored, ensure_ascii=False))
    parts.append(
        "The evaluation protocol is FROZEN by the runtime; do not set eval. "
        "Frozen eval (context only):\n" + json.dumps(eval_cfg or {}, ensure_ascii=False)
    )
    parts.append("Available data operators (name/kind/params):\n"
                 + json.dumps(_summarize_catalog(catalog), ensure_ascii=False))
    parts.append(f"Now return the PlanOut JSON with exactly {len(targets)} edges, one per slot in order.")
    return "\n\n".join(parts)


def next_plan(
    *,
    agent,
    targets: list[dict],
    instruct: str = "",
    goal: str = "",
    prev_review=None,
    explored: list | None = None,
    eval_cfg: dict | None = None,
    round_idx: int = 0,
    catalog: list[dict] | None = None,
) -> PlanOut:
    """Ask the planner for one edge per slot (typed PlanOut)."""
    task = _build_prompt(
        instruct=instruct, goal=goal, prev_review=prev_review, targets=targets,
        explored=explored or [], eval_cfg=eval_cfg, round_idx=round_idx, catalog=catalog,
    )
    return run_structured(agent, task, PlanOut)
