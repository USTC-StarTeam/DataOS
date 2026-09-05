"""Leader planning prompt.

Code sends a compact context bundle to the persistent leader dialog. The
agent returns exactly one :class:`ExperimentPlan`; code validates and applies it.
"""

from __future__ import annotations

import json

from dataos import prompt as _prompt
from dataos.agent.llm.structured import run_structured_dialog

from .schemas import ExperimentPlan


def _summarize_catalog(catalog: list[dict] | None, limit: int = 60) -> list[dict]:
    out = []
    for op in (catalog or [])[:limit]:
        row = {
            "name": op.get("name"),
            "kind": op.get("kind"),
            "mode": op.get("mode"),  # offline (data-prep) | online (training-time policy)
            "effect": op.get("effect"),  # one-line summary of what the operator does
            "dataset_backends": op.get("dataset_backends"),
            "params": [p.get("name") for p in (op.get("params") or [])],
        }
        out.append({k: v for k, v in row.items() if v is not None})
    return out


def _build_prompt(
    *,
    context: dict,
    catalog: list[dict] | None,
    validation_errors: list[str] | None = None,
    catalog_pointer: str | None = None,
    catalog_path: str = "",
    template: str = "leader/turn/plan",
) -> str:
    return _prompt.render(
        template,
        context_json=json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True),
        catalog_json=json.dumps(_summarize_catalog(catalog), ensure_ascii=False, indent=2),
        catalog_pointer=catalog_pointer,
        catalog_path=catalog_path,
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
    catalog_pointer: str | None = None,
    catalog_path: str = "",
    template: str = "leader/turn/plan",
) -> tuple[ExperimentPlan, str | None]:
    """Ask the leader dialog for one next experiment plan."""

    task = _build_prompt(
        context=context,
        catalog=catalog,
        validation_errors=validation_errors,
        catalog_pointer=catalog_pointer,
        catalog_path=catalog_path,
        template=template,
    )
    return run_structured_dialog(agent, task, ExperimentPlan, session_id=session_id)
