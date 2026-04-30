from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from dataos.research.store import ResearchStore


_OPEN_TASK_STATUSES = frozenset({"pending", "scheduled", "running"})


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    now: datetime
    project: Mapping[str, Any]
    project_status: str
    research_tasks: tuple[Mapping[str, Any], ...]
    agent_tasks: tuple[Mapping[str, Any], ...]
    proposed_proposals: tuple[Mapping[str, Any], ...]
    accepted_proposals: tuple[Mapping[str, Any], ...]
    all_proposals: tuple[Mapping[str, Any], ...]
    hypotheses: tuple[Mapping[str, Any], ...]
    candidate_operators: tuple[Mapping[str, Any], ...]
    recent_events: tuple[Mapping[str, Any], ...]
    experiment_run_budget: Mapping[str, Any]
    project_concurrency: Mapping[str, Any]
    queue_depth: int
    wall_clock_hours_limit: float | None


def build_snapshot(
    store: ResearchStore,
    project_id: str,
    now: datetime,
) -> StateSnapshot:
    state = store.get_project_state(project_id)
    research_tasks = tuple(state["tasks"])
    agent_tasks = tuple(state["agent_tasks"])
    all_proposals = tuple(state["agent_proposals"])
    proposed = tuple(p for p in all_proposals if p.get("status") == "proposed")
    accepted = tuple(p for p in all_proposals if p.get("status") == "accepted")
    queue_depth = sum(1 for task in research_tasks if task.get("status") in _OPEN_TASK_STATUSES)
    project = state["project"]
    budget = project.get("budget") or {}
    template_budget = (project.get("task_template") or {}).get("budget") or {}
    raw_wall = budget.get("wall_clock_hours")
    if raw_wall is None and isinstance(template_budget, dict):
        raw_wall = template_budget.get("wall_clock_hours")
    wall_clock_limit: float | None
    try:
        wall_clock_limit = float(raw_wall) if raw_wall is not None else None
    except (TypeError, ValueError):
        wall_clock_limit = None
    return StateSnapshot(
        now=now,
        project=project,
        project_status=str(project.get("status", "")),
        research_tasks=research_tasks,
        agent_tasks=agent_tasks,
        proposed_proposals=proposed,
        accepted_proposals=accepted,
        all_proposals=all_proposals,
        hypotheses=tuple(state["hypotheses"]),
        candidate_operators=tuple(state["candidate_operators"]),
        recent_events=tuple(state.get("events", [])[-200:]),
        experiment_run_budget=store.get_experiment_run_budget(project_id),
        project_concurrency=store.get_project_concurrency(project_id),
        queue_depth=queue_depth,
        wall_clock_hours_limit=wall_clock_limit,
    )
