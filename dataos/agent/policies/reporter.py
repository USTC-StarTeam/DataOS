from __future__ import annotations

from typing import Any, Mapping

from dataos.agent.actions import Action
from dataos.agent.snapshot import StateSnapshot


def decide(snapshot: StateSnapshot) -> list[Action]:
    project_id = str(snapshot.project["project_id"])
    if snapshot.project_status not in {"created", "running"}:
        return [Action(kind="skip", project_id=project_id, reason="project_inactive")]
    completed = [
        task for task in snapshot.research_tasks if task.get("status") == "completed"
    ]
    if not completed:
        return [Action(kind="skip", project_id=project_id, reason="reporter_no_completed")]
    if any(p.get("proposal_type") == "report" for p in snapshot.all_proposals):
        return [
            Action(kind="skip", project_id=project_id, reason="reporter_already_proposed")
        ]
    open_research = [
        task
        for task in snapshot.research_tasks
        if task.get("status") in {"pending", "blocked", "scheduled", "running"}
    ]
    if open_research:
        return [Action(kind="skip", project_id=project_id, reason="reporter_open_tasks")]
    if _open_agent_tasks_for_scope(snapshot, "final_report", ("report",)):
        return [Action(kind="skip", project_id=project_id, reason="reporter_inflight")]
    if not (len(completed) >= 2 or _budget_exhausted(snapshot)):
        return [
            Action(kind="skip", project_id=project_id, reason="reporter_threshold_not_met")
        ]
    # Wall-clock check is W2: read snapshot.wall_clock_hours_limit and act on it.
    return [
        Action(
            kind="spawn_specialists",
            project_id=project_id,
            roles=("report",),
            context_scope="final_report",
            priority=7,
        )
    ]


def _budget_exhausted(snapshot: StateSnapshot) -> bool:
    remaining = snapshot.experiment_run_budget.get("remaining")
    return remaining is not None and remaining == 0


def _open_agent_tasks_for_scope(
    snapshot: StateSnapshot,
    context_scope: str,
    roles: tuple[str, ...],
) -> list[Mapping[str, Any]]:
    role_set = set(roles)
    open_statuses = {"pending", "running", "failed"}
    matches: list[Mapping[str, Any]] = []
    for task in snapshot.agent_tasks:
        if task.get("status") not in open_statuses:
            continue
        if role_set and task.get("role") not in role_set:
            continue
        ctx = task.get("input_context")
        if isinstance(ctx, dict) and ctx.get("context_scope") == context_scope:
            matches.append(task)
    return matches
