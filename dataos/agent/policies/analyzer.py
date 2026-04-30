from __future__ import annotations

from typing import Any, Mapping

from dataos.agent.actions import Action
from dataos.agent.snapshot import StateSnapshot


RESULT_AGENT_ROLES: tuple[str, ...] = (
    "evaluation_analyst",
    "critic",
    "operator_designer",
)


def decide(snapshot: StateSnapshot) -> list[Action]:
    project_id = str(snapshot.project["project_id"])
    if snapshot.project_status not in {"created", "running"}:
        return [Action(kind="skip", project_id=project_id, reason="project_inactive")]
    if not _has_uncovered_completed_tasks(snapshot):
        return [Action(kind="skip", project_id=project_id, reason="analyzer_no_new_results")]
    if _open_agent_tasks_for_scope(snapshot, "result_analysis", RESULT_AGENT_ROLES):
        return [Action(kind="skip", project_id=project_id, reason="analyzer_inflight")]
    return [
        Action(
            kind="spawn_specialists",
            project_id=project_id,
            roles=RESULT_AGENT_ROLES,
            context_scope="result_analysis",
            priority=8,
        )
    ]


def _has_uncovered_completed_tasks(snapshot: StateSnapshot) -> bool:
    completed = [task for task in snapshot.research_tasks if task.get("status") == "completed"]
    if not completed:
        return False
    covered: set[str] = set()
    for event in snapshot.recent_events:
        if event.get("action") != "analysis_recorded":
            continue
        payload = event.get("payload", {})
        if isinstance(payload, dict):
            covered.update(str(task_id) for task_id in payload.get("covered_task_ids", []))
    return any(str(task["task_id"]) not in covered for task in completed)


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
