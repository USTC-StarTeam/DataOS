"""Main-agent planning prompt.

Code sends a compact context bundle to the persistent main-agent dialog. The
agent returns exactly one :class:`ExperimentPlan`; code validates and applies it.
"""

from __future__ import annotations

import json

from dataos import prompt as _prompt
from dataos.agent.llm.structured import run_structured_dialog

from .schemas import ExperimentPlan


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


def _build_prompt(
    *,
    context: dict,
    catalog: list[dict] | None,
    validation_errors: list[str] | None = None,
) -> str:
    return _prompt.render(
        "planner_task",
        context_json=json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True),
        catalog_json=json.dumps(_summarize_catalog(catalog), ensure_ascii=False, indent=2),
        validation_errors_json=(
            json.dumps(validation_errors, ensure_ascii=False, indent=2)
            if validation_errors
            else None
        ),
    )


def next_plan(
    *,
    agent,
    context: dict,
    catalog: list[dict] | None = None,
    session_id: str | None = None,
    validation_errors: list[str] | None = None,
) -> tuple[ExperimentPlan, str | None]:
    """Ask the main-agent dialog for one next experiment plan."""

    task = _build_prompt(
        context=context,
        catalog=catalog,
        validation_errors=validation_errors,
    )
    return run_structured_dialog(agent, task, ExperimentPlan, session_id=session_id)
