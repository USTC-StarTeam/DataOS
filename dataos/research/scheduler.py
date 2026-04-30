from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dataos.research.models import ResearchTask, Resource
from dataos.research.store import ResearchStore


@dataclass(slots=True)
class SchedulerReport:
    scheduled_task_ids: list[str] = field(default_factory=list)
    unblocked_task_ids: list[str] = field(default_factory=list)
    blocked_task_ids: list[str] = field(default_factory=list)
    skipped_task_ids: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "scheduled_task_ids": self.scheduled_task_ids,
            "unblocked_task_ids": self.unblocked_task_ids,
            "blocked_task_ids": self.blocked_task_ids,
            "skipped_task_ids": self.skipped_task_ids,
            "errors": self.errors,
        }


@dataclass(slots=True)
class RecoveryReport:
    requeued_task_ids: list[str] = field(default_factory=list)
    skipped_task_ids: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "requeued_task_ids": self.requeued_task_ids,
            "skipped_task_ids": self.skipped_task_ids,
            "errors": self.errors,
        }


def scheduler_tick(
    workspace_dir: str | Path,
    *,
    max_tasks: int | None = None,
    project_id: str | None = None,
) -> SchedulerReport:
    store = ResearchStore(workspace_dir)
    store.prepare()
    _ensure_default_local_resource(store)
    report = SchedulerReport()
    report.unblocked_task_ids = [
        task.task_id
        for task in store.unblock_recoverable_tasks(project_id=project_id, actor="scheduler")
    ]
    report.blocked_task_ids = [
        task.task_id
        for task in store.block_unrunnable_tasks(project_id=project_id, actor="scheduler")
    ]
    for task in store.list_ready_tasks(project_id=project_id):
        if max_tasks is not None and len(report.scheduled_task_ids) >= max_tasks:
            break
        try:
            if _project_concurrency_exhausted(store, task.project_id):
                report.skipped_task_ids.append(task.task_id)
                continue
            resource = _select_resource(store, task)
            if resource is None:
                report.skipped_task_ids.append(task.task_id)
                continue
            store.schedule_research_task(task, resource, actor="scheduler")
            report.scheduled_task_ids.append(task.task_id)
        except Exception as exc:  # pragma: no cover - defensive scheduler isolation
            report.errors[task.task_id] = str(exc)
    return report


def recover_stale_tasks(
    workspace_dir: str | Path,
    *,
    older_than_seconds: int,
    project_id: str | None = None,
    statuses: list[str] | None = None,
) -> RecoveryReport:
    store = ResearchStore(workspace_dir)
    store.prepare()
    cutoff = datetime.now(UTC) - timedelta(seconds=max(0, int(older_than_seconds)))
    target_statuses = statuses or ["scheduled", "running"]
    report = RecoveryReport()
    for status in target_statuses:
        if status not in {"scheduled", "running"}:
            raise ValueError(f"Unsupported stale recovery status: {status}")
        for task in store.list_research_tasks(project_id, status=status):
            try:
                updated_at = datetime.fromisoformat(task.updated_at)
                if updated_at > cutoff:
                    report.skipped_task_ids.append(task.task_id)
                    continue
                store.requeue_research_task(
                    task.task_id,
                    actor="scheduler",
                    reason=(
                        f"Recovered stale {task.status} task older than "
                        f"{older_than_seconds} seconds."
                    ),
                )
                report.requeued_task_ids.append(task.task_id)
            except Exception as exc:  # pragma: no cover - defensive scheduler isolation
                report.errors[task.task_id] = str(exc)
    return report


def _project_concurrency_exhausted(store: ResearchStore, project_id: str) -> bool:
    concurrency = store.get_project_concurrency(project_id)
    remaining = concurrency["remaining"]
    return remaining is not None and remaining <= 0


def _ensure_default_local_resource(store: ResearchStore) -> None:
    if store.list_resources():
        return
    store.add_resource(
        resource_id="local-default",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=2,
        tags=["default"],
    )


def _select_resource(store: ResearchStore, task: ResearchTask) -> Resource | None:
    for resource in store.list_resources():
        if not _resource_matches(task, resource):
            continue
        if store.active_lease_count(resource.resource_id) >= resource.max_concurrent_tasks:
            continue
        return resource
    return None


def _resource_matches(task: ResearchTask, resource: Resource) -> bool:
    request = task.resource_request
    required_gpus = int(request.get("gpus", 0) or 0)
    if required_gpus > len(resource.gpu_ids):
        return False
    requested_host = request.get("host")
    if requested_host is not None and requested_host != resource.host:
        return False
    requested_tags = set(request.get("tags", []))
    return requested_tags.issubset(set(resource.tags))
