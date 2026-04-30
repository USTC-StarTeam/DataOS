from __future__ import annotations

from typing import Any

from dataos.research.store import ResearchStore


def append_loop_tick(
    store: ResearchStore,
    *,
    project_id: str,
    loop_name: str,
    actions: list[dict[str, Any]],
    queue_depth: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "actions": actions,
        "queue_depth": queue_depth,
    }
    if extra:
        payload.update(extra)
    store.append_event(
        project_id=project_id,
        actor=f"loop:{loop_name}",
        action="loop_tick",
        subject_type="agent_loop",
        subject_id=loop_name,
        payload=payload,
    )


def append_specialist_call(
    store: ResearchStore,
    *,
    project_id: str,
    agent_task_id: str,
    role: str,
    phase: str,
    priority: int,
    context_scope: str | None = None,
    error: str | None = None,
    kind: str = "specialist",
) -> None:
    payload: dict[str, Any] = {
        "phase": phase,
        "role": role,
        "priority": priority,
        "kind": kind,
    }
    if context_scope is not None:
        payload["context_scope"] = context_scope
    if error is not None:
        payload["error"] = error
    store.append_event(
        project_id=project_id,
        actor=f"specialist:{role}",
        action="specialist_call",
        subject_type="agent_task",
        subject_id=agent_task_id,
        payload=payload,
    )


def append_recovery_action(
    store: ResearchStore,
    *,
    project_id: str | None,
    requeued_task_ids: list[str],
    older_than_seconds: int,
) -> None:
    store.append_event(
        project_id=project_id,
        actor="daemon:supervisor",
        action="recovery_action",
        subject_type="research_task",
        subject_id="*",
        payload={
            "requeued_task_ids": requeued_task_ids,
            "older_than_seconds": older_than_seconds,
        },
    )


def append_backpressure_triggered(
    store: ResearchStore,
    *,
    project_id: str,
    layer: str,
    reason: str,
    payload: dict[str, Any] | None = None,
) -> None:
    body: dict[str, Any] = {"layer": layer, "reason": reason}
    if payload:
        body.update(payload)
    store.append_event(
        project_id=project_id,
        actor="daemon:supervisor",
        action="backpressure_triggered",
        subject_type="agent_loop",
        subject_id=layer,
        payload=body,
    )


def append_daemon_lifecycle(
    store: ResearchStore,
    *,
    project_id: str,
    phase: str,
    payload: dict[str, Any] | None = None,
) -> None:
    store.append_event(
        project_id=project_id,
        actor="daemon:supervisor",
        action=f"daemon_{phase}",
        subject_type="daemon",
        subject_id=project_id,
        payload=payload or {},
    )
