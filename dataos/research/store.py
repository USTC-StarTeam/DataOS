from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataos.core.artifact import utc_now_iso
from dataos.core.task import TaskTemplate
from dataos.research.models import (
    AGENT_ROLES,
    AGENT_TASK_STATUSES,
    ACTIVE_PROJECT_STATUSES,
    CANDIDATE_OPERATOR_STATUSES,
    PROPOSAL_STATUSES,
    PROJECT_STATUSES,
    TASK_STATUSES,
    TASK_TYPES,
    TERMINAL_AGENT_TASK_STATUSES,
    TERMINAL_PROPOSAL_STATUSES,
    TERMINAL_TASK_STATUSES,
    AgentProposal,
    AgentTask,
    CandidateOperator,
    Event,
    Hypothesis,
    ResearchProject,
    ResearchTask,
    Resource,
    ResourceLease,
    new_agent_task_id,
    new_candidate_operator_id,
    new_event_id,
    new_hypothesis_id,
    new_lease_id,
    new_project_id,
    new_proposal_id,
    new_task_id,
)


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _loads(value: str | None, default: Any) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value)


def _placeholders(items: Any) -> str:
    return ", ".join("?" for _ in items)


@dataclass(slots=True)
class ResearchStore:
    workspace_dir: Path | str
    db_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir)
        self.db_path = self.workspace_dir / "dataos.db"

    def prepare(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    dataset_path TEXT NOT NULL,
                    task_template_json TEXT NOT NULL,
                    budget_json TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_tasks (
                    task_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    resource_request_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    error TEXT,
                    lease_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );

                CREATE TABLE IF NOT EXISTS task_dependencies (
                    task_id TEXT NOT NULL,
                    depends_on_task_id TEXT NOT NULL,
                    PRIMARY KEY (task_id, depends_on_task_id)
                );

                CREATE TABLE IF NOT EXISTS candidate_operators (
                    candidate_operator_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    hypothesis_id TEXT,
                    name TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    validation_report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resources (
                    resource_id TEXT PRIMARY KEY,
                    host TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    gpu_ids_json TEXT NOT NULL,
                    max_concurrent_tasks INTEGER NOT NULL,
                    tags_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resource_leases (
                    lease_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    host TEXT NOT NULL,
                    gpu_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    released_at TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_tasks (
                    agent_task_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    input_context_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    agent_task_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    proposal_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_project(
        self,
        *,
        goal: str,
        dataset_path: str | Path,
        task_template: TaskTemplate | dict[str, Any],
        budget: dict[str, Any],
        constraints: dict[str, Any] | None = None,
    ) -> ResearchProject:
        self.prepare()
        task_record = (
            task_template.to_record()
            if isinstance(task_template, TaskTemplate)
            else dict(task_template)
        )
        now = utc_now_iso()
        project = ResearchProject(
            project_id=new_project_id(),
            goal=goal,
            dataset_path=str(dataset_path),
            task_template=task_record,
            budget=dict(budget),
            constraints=dict(constraints or {}),
            created_at=now,
            updated_at=now,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    project_id, goal, dataset_path, task_template_json, budget_json,
                    constraints_json, status, summary_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.project_id,
                    project.goal,
                    project.dataset_path,
                    _dumps(project.task_template),
                    _dumps(project.budget),
                    _dumps(project.constraints),
                    project.status,
                    _dumps(project.summary),
                    project.created_at,
                    project.updated_at,
                ),
            )
        self.append_event(
            project_id=project.project_id,
            actor="system",
            action="project_created",
            subject_type="project",
            subject_id=project.project_id,
            payload=project.to_record(),
        )
        return project

    def get_project(self, project_id: str) -> ResearchProject:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        return self._project_from_row(row)

    def update_project_summary(
        self,
        project_id: str,
        summary: dict[str, Any],
        *,
        status: str | None = None,
        actor: str = "dataos_agent",
    ) -> ResearchProject:
        project = self.get_project(project_id)
        now = utc_now_iso()
        next_status = project.status if status is None else status
        if next_status not in PROJECT_STATUSES:
            raise ValueError(f"Unsupported project status: {next_status}")
        status_changed = next_status != project.status
        self._validate_project_status_transition(project.status, next_status)
        if status_changed and next_status == "completed":
            self._validate_project_can_complete(project_id)
        if status_changed and next_status == "paused":
            self._pause_scheduled_project_work(project_id, actor=actor)
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE projects
                SET summary_json = ?, status = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (_dumps(summary), next_status, now, project_id),
            )
        updated = self.get_project(project_id)
        if updated.project_id != project_id:
            raise KeyError(f"Unknown project_id: {project_id}")
        self.append_event(
            project_id=project_id,
            actor=actor,
            action="project_summary_updated",
            subject_type="project",
            subject_id=project_id,
            payload=updated.to_record(),
        )
        if status_changed:
            self.append_event(
                project_id=project_id,
                actor=actor,
                action=f"project_{next_status}",
                subject_type="project",
                subject_id=project_id,
                payload=updated.to_record(),
            )
            if next_status == "cancelled":
                self._cancel_open_project_work(project_id, actor=actor)
        return updated

    def update_project_status(
        self,
        project_id: str,
        status: str,
        *,
        actor: str = "human",
    ) -> ResearchProject:
        if status not in PROJECT_STATUSES:
            raise ValueError(f"Unsupported project status: {status}")
        project = self.get_project(project_id)
        self._validate_project_status_transition(project.status, status)
        if status == "completed":
            self._validate_project_can_complete(project_id)
        if project.status != status and status == "paused":
            self._pause_scheduled_project_work(project_id, actor=actor)
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE projects
                SET status = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (status, now, project_id),
            )
        updated = self.get_project(project_id)
        self.append_event(
            project_id=project_id,
            actor=actor,
            action=f"project_{status}",
            subject_type="project",
            subject_id=project_id,
            payload=updated.to_record(),
        )
        if status == "cancelled":
            self._cancel_open_project_work(project_id, actor=actor)
        return updated

    def mark_project_running(
        self,
        project_id: str,
        *,
        actor: str = "system",
    ) -> ResearchProject:
        project = self.get_project(project_id)
        if project.status != "created":
            return project
        return self.update_project_status(project_id, "running", actor=actor)

    def require_active_project(self, project_id: str) -> ResearchProject:
        project = self.get_project(project_id)
        if project.status not in ACTIVE_PROJECT_STATUSES:
            raise ValueError(f"Project is not active: {project_id} ({project.status})")
        return project

    def add_hypothesis(
        self,
        project_id: str,
        *,
        title: str,
        rationale: str,
        status: str = "active",
        created_by: str = "dataos_agent",
        metadata: dict[str, Any] | None = None,
    ) -> Hypothesis:
        self.require_active_project(project_id)
        now = utc_now_iso()
        hypothesis = Hypothesis(
            hypothesis_id=new_hypothesis_id(),
            project_id=project_id,
            title=title,
            rationale=rationale,
            status=status,
            created_by=created_by,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO hypotheses (
                    hypothesis_id, project_id, title, rationale, status, created_by,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis.hypothesis_id,
                    hypothesis.project_id,
                    hypothesis.title,
                    hypothesis.rationale,
                    hypothesis.status,
                    hypothesis.created_by,
                    _dumps(hypothesis.metadata),
                    hypothesis.created_at,
                    hypothesis.updated_at,
                ),
            )
        self.mark_project_running(project_id, actor=created_by)
        self.append_event(
            project_id=project_id,
            actor=created_by,
            action="hypothesis_created",
            subject_type="hypothesis",
            subject_id=hypothesis.hypothesis_id,
            payload=hypothesis.to_record(),
        )
        return hypothesis

    def list_hypotheses(self, project_id: str) -> list[Hypothesis]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE project_id = ? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        return [self._hypothesis_from_row(row) for row in rows]

    def submit_research_task(
        self,
        project_id: str,
        *,
        task_type: str,
        spec: dict[str, Any],
        dependencies: list[str] | None = None,
        actor: str = "human",
        priority: int = 0,
        resource_request: dict[str, Any] | None = None,
    ) -> ResearchTask:
        task_dependencies = list(dependencies or [])
        self.validate_research_task_request(
            project_id,
            task_type=task_type,
            spec=spec,
            dependencies=task_dependencies,
        )
        now = utc_now_iso()
        task = ResearchTask(
            task_id=new_task_id(),
            project_id=project_id,
            task_type=task_type,
            spec=dict(spec),
            actor=actor,
            priority=int(priority),
            dependencies=task_dependencies,
            resource_request=dict(resource_request or spec.get("resource_request", {})),
            created_at=now,
            updated_at=now,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO research_tasks (
                    task_id, project_id, task_type, spec_json, status, actor, priority,
                    resource_request_json, result_json, error, lease_id, created_at,
                    updated_at, started_at, finished_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.project_id,
                    task.task_type,
                    _dumps(task.spec),
                    task.status,
                    task.actor,
                    task.priority,
                    _dumps(task.resource_request),
                    _dumps(task.result),
                    task.error,
                    task.lease_id,
                    task.created_at,
                    task.updated_at,
                    task.started_at,
                    task.finished_at,
                ),
            )
            for dependency_id in task.dependencies:
                conn.execute(
                    """
                    INSERT INTO task_dependencies (task_id, depends_on_task_id)
                    VALUES (?, ?)
                    """,
                    (task.task_id, dependency_id),
                )
        self.mark_project_running(project_id, actor=actor)
        self.append_event(
            project_id=project_id,
            actor=actor,
            action="research_task_submitted",
            subject_type="research_task",
            subject_id=task.task_id,
            payload=task.to_record(),
        )
        return task

    def validate_research_task_request(
        self,
        project_id: str,
        *,
        task_type: str,
        spec: dict[str, Any],
        dependencies: list[str] | None = None,
        reserved_experiment_runs: int = 0,
    ) -> None:
        project = self.require_active_project(project_id)
        self._validate_task_type(task_type)
        self._validate_operator_refs(spec)
        self._validate_task_dependencies(project_id, list(dependencies or []))
        self._validate_project_budget(
            project,
            task_type,
            reserved_experiment_runs=reserved_experiment_runs,
        )

    def get_research_task(self, task_id: str) -> ResearchTask:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            dependencies = self._load_dependencies(conn, task_id)
        if row is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return self._research_task_from_row(row, dependencies)

    def list_research_tasks(
        self,
        project_id: str | None = None,
        *,
        status: str | None = None,
        task_type: str | None = None,
    ) -> list[ResearchTask]:
        query = "SELECT * FROM research_tasks WHERE 1 = 1"
        params: list[Any] = []
        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if task_type is not None:
            query += " AND task_type = ?"
            params.append(task_type)
        query += " ORDER BY priority DESC, created_at ASC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                self._research_task_from_row(row, self._load_dependencies(conn, row["task_id"]))
                for row in rows
            ]

    def list_ready_tasks(
        self,
        limit: int | None = None,
        *,
        project_id: str | None = None,
    ) -> list[ResearchTask]:
        query = """
            SELECT tasks.*
            FROM research_tasks AS tasks
            JOIN projects AS projects ON tasks.project_id = projects.project_id
            WHERE tasks.status = 'pending'
        """
        params: list[Any] = []
        if project_id is not None:
            query += " AND tasks.project_id = ?"
            params.append(project_id)
        query += f" AND projects.status IN ({_placeholders(ACTIVE_PROJECT_STATUSES)})"
        params.extend(sorted(ACTIVE_PROJECT_STATUSES))
        query += " ORDER BY tasks.priority DESC, tasks.created_at ASC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            tasks = [
                self._research_task_from_row(row, self._load_dependencies(conn, row["task_id"]))
                for row in rows
            ]
            ready = [task for task in tasks if self._dependencies_completed(conn, task)]
        return ready if limit is None else ready[:limit]

    def block_unrunnable_tasks(
        self,
        project_id: str | None = None,
        *,
        actor: str = "scheduler",
    ) -> list[ResearchTask]:
        blocked: list[ResearchTask] = []
        while True:
            newly_blocked: list[ResearchTask] = []
            for task in self.list_research_tasks(project_id, status="pending"):
                if self.get_project(task.project_id).status not in ACTIVE_PROJECT_STATUSES:
                    continue
                blocking_dependencies = self._blocking_dependencies(task)
                if not blocking_dependencies:
                    continue
                error = "Blocked by dependency status: " + ", ".join(
                    f"{dependency_id}={status}"
                    for dependency_id, status in blocking_dependencies.items()
                )
                newly_blocked.append(
                    self.update_task_status(
                        task.task_id,
                        "blocked",
                        actor=actor,
                        error=error,
                    )
                )
            if not newly_blocked:
                return blocked
            blocked.extend(newly_blocked)

    def unblock_recoverable_tasks(
        self,
        project_id: str | None = None,
        *,
        actor: str = "scheduler",
    ) -> list[ResearchTask]:
        unblocked: list[ResearchTask] = []
        while True:
            newly_unblocked: list[ResearchTask] = []
            for task in self.list_research_tasks(project_id, status="blocked"):
                if self.get_project(task.project_id).status not in ACTIVE_PROJECT_STATUSES:
                    continue
                if self._blocking_dependencies(task):
                    continue
                newly_unblocked.append(
                    self.update_task_status(
                        task.task_id,
                        "pending",
                        actor=actor,
                        error=None,
                    )
                )
            if not newly_unblocked:
                return unblocked
            unblocked.extend(newly_unblocked)

    def update_task_status(
        self,
        task_id: str,
        status: str,
        *,
        actor: str = "system",
        result: dict[str, Any] | None = None,
        error: str | None = None,
        lease_id: str | None = None,
    ) -> ResearchTask:
        if status not in TASK_STATUSES:
            raise ValueError(f"Unsupported task status: {status}")
        task = self.get_research_task(task_id)
        self._validate_task_status_transition(task.status, status)
        now = utc_now_iso()
        started_at = task.started_at
        finished_at = task.finished_at
        if status == "running" and started_at is None:
            started_at = now
        if status in {"completed", "failed", "cancelled"}:
            finished_at = now
        merged_result = task.result if result is None else dict(result)
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE research_tasks
                SET status = ?, result_json = ?, error = ?, lease_id = ?, updated_at = ?,
                    started_at = ?, finished_at = ?
                WHERE task_id = ?
                """,
                (
                    status,
                    _dumps(merged_result),
                    error,
                    lease_id if lease_id is not None else task.lease_id,
                    now,
                    started_at,
                    finished_at,
                    task_id,
                ),
            )
        updated = self.get_research_task(task_id)
        self.append_event(
            project_id=updated.project_id,
            actor=actor,
            action=f"research_task_{status}",
            subject_type="research_task",
            subject_id=task_id,
            payload=updated.to_record(),
        )
        return updated

    def cancel_research_task(
        self,
        task_id: str,
        *,
        actor: str = "human",
        reason: str | None = None,
    ) -> ResearchTask:
        task = self.get_research_task(task_id)
        if task.status == "cancelled":
            return task
        if task.status in TERMINAL_TASK_STATUSES:
            raise ValueError(f"Cannot cancel terminal task: {task_id}")
        updated = self.update_task_status(
            task_id,
            "cancelled",
            actor=actor,
            error=reason,
            lease_id=task.lease_id,
        )
        if task.lease_id is not None:
            self.release_lease(task.lease_id)
        return updated

    def requeue_research_task(
        self,
        task_id: str,
        *,
        actor: str = "human",
        reason: str | None = None,
    ) -> ResearchTask:
        task = self.get_research_task(task_id)
        if task.status == "completed":
            raise ValueError(f"Cannot requeue completed task: {task_id}")
        self.require_active_project(task.project_id)
        if task.status == "pending":
            return task
        if task.status == "cancelled":
            self._validate_project_budget(self.get_project(task.project_id), task.task_type)
        if task.lease_id is not None:
            self.release_lease(task.lease_id)
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE research_tasks
                SET status = 'pending', result_json = ?, error = ?, lease_id = NULL,
                    updated_at = ?, started_at = NULL, finished_at = NULL
                WHERE task_id = ?
                """,
                (_dumps({}), reason, now, task_id),
            )
        updated = self.get_research_task(task_id)
        self.append_event(
            project_id=updated.project_id,
            actor=actor,
            action="research_task_requeued",
            subject_type="research_task",
            subject_id=task_id,
            payload=updated.to_record(),
        )
        return updated

    def add_resource(
        self,
        *,
        resource_id: str,
        host: str,
        kind: str = "local",
        gpu_ids: list[str] | None = None,
        max_concurrent_tasks: int = 1,
        tags: list[str] | None = None,
    ) -> Resource:
        now = utc_now_iso()
        resource = Resource(
            resource_id=resource_id,
            host=host,
            kind=kind,
            gpu_ids=list(gpu_ids or []),
            max_concurrent_tasks=max(1, int(max_concurrent_tasks)),
            tags=list(tags or []),
            created_at=now,
            updated_at=now,
        )
        self.prepare()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO resources (
                    resource_id, host, kind, gpu_ids_json, max_concurrent_tasks,
                    tags_json, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resource.resource_id,
                    resource.host,
                    resource.kind,
                    _dumps(resource.gpu_ids),
                    resource.max_concurrent_tasks,
                    _dumps(resource.tags),
                    resource.status,
                    resource.created_at,
                    resource.updated_at,
                ),
            )
        return resource

    def list_resources(self) -> list[Resource]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM resources WHERE status = 'active' ORDER BY resource_id"
            ).fetchall()
        return [self._resource_from_row(row) for row in rows]

    def create_lease(self, task: ResearchTask, resource: Resource) -> ResourceLease:
        gpu_count = int(task.resource_request.get("gpus", 0) or 0)
        gpu_ids = resource.gpu_ids[:gpu_count] if gpu_count > 0 else []
        lease = ResourceLease(
            lease_id=new_lease_id(),
            task_id=task.task_id,
            resource_id=resource.resource_id,
            host=resource.host,
            gpu_ids=gpu_ids,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO resource_leases (
                    lease_id, task_id, resource_id, host, gpu_ids_json, status,
                    created_at, released_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lease.lease_id,
                    lease.task_id,
                    lease.resource_id,
                    lease.host,
                    _dumps(lease.gpu_ids),
                    lease.status,
                    lease.created_at,
                    lease.released_at,
                ),
            )
        return lease

    def schedule_research_task(
        self,
        task: ResearchTask,
        resource: Resource,
        *,
        actor: str = "scheduler",
    ) -> ResearchTask:
        self.require_active_project(task.project_id)
        gpu_count = int(task.resource_request.get("gpus", 0) or 0)
        gpu_ids = resource.gpu_ids[:gpu_count] if gpu_count > 0 else []
        lease = ResourceLease(
            lease_id=new_lease_id(),
            task_id=task.task_id,
            resource_id=resource.resource_id,
            host=resource.host,
            gpu_ids=gpu_ids,
        )
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO resource_leases (
                    lease_id, task_id, resource_id, host, gpu_ids_json, status,
                    created_at, released_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lease.lease_id,
                    lease.task_id,
                    lease.resource_id,
                    lease.host,
                    _dumps(lease.gpu_ids),
                    lease.status,
                    lease.created_at,
                    lease.released_at,
                ),
            )
            cursor = conn.execute(
                """
                UPDATE research_tasks
                SET status = 'scheduled', lease_id = ?, updated_at = ?
                WHERE task_id = ? AND status = 'pending' AND lease_id IS NULL
                """,
                (lease.lease_id, now, task.task_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Task is no longer schedulable: {task.task_id}")
        updated = self.get_research_task(task.task_id)
        self.mark_project_running(updated.project_id, actor=actor)
        self.append_event(
            project_id=updated.project_id,
            actor=actor,
            action="research_task_scheduled",
            subject_type="research_task",
            subject_id=task.task_id,
            payload=updated.to_record(),
        )
        return updated

    def get_lease(self, lease_id: str) -> ResourceLease:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM resource_leases WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        return self._lease_from_row(row)

    def list_leases(
        self,
        *,
        status: str | None = None,
        resource_id: str | None = None,
        project_id: str | None = None,
    ) -> list[ResourceLease]:
        query = "SELECT leases.* FROM resource_leases AS leases"
        params: list[Any] = []
        if project_id is not None:
            query += " JOIN research_tasks AS tasks ON leases.task_id = tasks.task_id"
        query += " WHERE 1 = 1"
        if status is not None:
            query += " AND leases.status = ?"
            params.append(status)
        if resource_id is not None:
            query += " AND leases.resource_id = ?"
            params.append(resource_id)
        if project_id is not None:
            query += " AND tasks.project_id = ?"
            params.append(project_id)
        query += " ORDER BY leases.created_at ASC, leases.lease_id ASC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._lease_from_row(row) for row in rows]

    def release_lease(self, lease_id: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE resource_leases
                SET status = 'released', released_at = ?
                WHERE lease_id = ? AND status = 'active'
                """,
                (now, lease_id),
            )

    def active_lease_count(self, resource_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM resource_leases
                WHERE resource_id = ? AND status = 'active'
                """,
                (resource_id,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def create_candidate_operator(
        self,
        project_id: str,
        *,
        name: str,
        spec: dict[str, Any],
        hypothesis_id: str | None = None,
        status: str = "proposed",
        actor: str = "dataos_agent",
    ) -> CandidateOperator:
        if status not in CANDIDATE_OPERATOR_STATUSES:
            raise ValueError(f"Unsupported candidate operator status: {status}")
        self.require_active_project(project_id)
        now = utc_now_iso()
        candidate = CandidateOperator(
            candidate_operator_id=new_candidate_operator_id(),
            project_id=project_id,
            hypothesis_id=hypothesis_id,
            name=name,
            spec=dict(spec),
            status=status,
            created_at=now,
            updated_at=now,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO candidate_operators (
                    candidate_operator_id, project_id, hypothesis_id, name, spec_json,
                    status, validation_report_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_operator_id,
                    candidate.project_id,
                    candidate.hypothesis_id,
                    candidate.name,
                    _dumps(candidate.spec),
                    candidate.status,
                    _dumps(candidate.validation_report),
                    candidate.created_at,
                    candidate.updated_at,
                ),
            )
        self.mark_project_running(project_id, actor=actor)
        self.append_event(
            project_id=project_id,
            actor=actor,
            action="candidate_operator_created",
            subject_type="candidate_operator",
            subject_id=candidate.candidate_operator_id,
            payload=candidate.to_record(),
        )
        return candidate

    def get_candidate_operator(self, candidate_operator_id: str) -> CandidateOperator:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM candidate_operators
                WHERE candidate_operator_id = ?
                """,
                (candidate_operator_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown candidate_operator_id: {candidate_operator_id}")
        return self._candidate_from_row(row)

    def update_candidate_operator_status(
        self,
        candidate_operator_id: str,
        status: str,
        *,
        validation_report: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> CandidateOperator:
        if status not in CANDIDATE_OPERATOR_STATUSES:
            raise ValueError(f"Unsupported candidate operator status: {status}")
        candidate = self.get_candidate_operator(candidate_operator_id)
        report = candidate.validation_report if validation_report is None else validation_report
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE candidate_operators
                SET status = ?, validation_report_json = ?, updated_at = ?
                WHERE candidate_operator_id = ?
                """,
                (status, _dumps(report), now, candidate_operator_id),
            )
        updated = self.get_candidate_operator(candidate_operator_id)
        self.append_event(
            project_id=updated.project_id,
            actor=actor,
            action=f"candidate_operator_{status}",
            subject_type="candidate_operator",
            subject_id=candidate_operator_id,
            payload=updated.to_record(),
        )
        return updated

    def list_candidate_operators(
        self,
        project_id: str,
        *,
        status: str | None = None,
    ) -> list[CandidateOperator]:
        query = "SELECT * FROM candidate_operators WHERE project_id = ?"
        params: list[Any] = [project_id]
        if status is not None:
            if status not in CANDIDATE_OPERATOR_STATUSES:
                raise ValueError(f"Unsupported candidate operator status: {status}")
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at ASC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def create_agent_task(
        self,
        project_id: str,
        *,
        role: str,
        input_context: dict[str, Any],
    ) -> AgentTask:
        if role not in AGENT_ROLES:
            raise ValueError(f"Unsupported agent role: {role}")
        self.require_active_project(project_id)
        now = utc_now_iso()
        task = AgentTask(
            agent_task_id=new_agent_task_id(),
            project_id=project_id,
            role=role,
            input_context=dict(input_context),
            created_at=now,
            updated_at=now,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_tasks (
                    agent_task_id, project_id, role, input_context_json, status,
                    output_json, error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.agent_task_id,
                    task.project_id,
                    task.role,
                    _dumps(task.input_context),
                    task.status,
                    _dumps(task.output),
                    task.error,
                    task.created_at,
                    task.updated_at,
                ),
            )
        self.mark_project_running(project_id, actor="research_manager")
        self.append_event(
            project_id=project_id,
            actor="research_manager",
            action="agent_task_created",
            subject_type="agent_task",
            subject_id=task.agent_task_id,
            payload=task.to_record(),
        )
        return task

    def get_agent_task(self, agent_task_id: str) -> AgentTask:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_tasks WHERE agent_task_id = ?",
                (agent_task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown agent_task_id: {agent_task_id}")
        return self._agent_task_from_row(row)

    def list_agent_tasks(
        self,
        project_id: str,
        *,
        status: str | None = None,
        role: str | None = None,
    ) -> list[AgentTask]:
        query = "SELECT * FROM agent_tasks WHERE project_id = ?"
        params: list[Any] = [project_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if role is not None:
            query += " AND role = ?"
            params.append(role)
        query += " ORDER BY created_at ASC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._agent_task_from_row(row) for row in rows]

    def update_agent_task_status(
        self,
        agent_task_id: str,
        status: str,
        *,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AgentTask:
        if status not in AGENT_TASK_STATUSES:
            raise ValueError(f"Unsupported agent task status: {status}")
        task = self.get_agent_task(agent_task_id)
        self._validate_agent_task_status_transition(task.status, status)
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE agent_tasks
                SET status = ?, output_json = ?, error = ?, updated_at = ?
                WHERE agent_task_id = ?
                """,
                (
                    status,
                    _dumps(task.output if output is None else output),
                    error,
                    now,
                    agent_task_id,
                ),
            )
        return self.get_agent_task(agent_task_id)

    def create_agent_proposal(
        self,
        *,
        project_id: str,
        agent_task_id: str,
        role: str,
        proposal_type: str,
        payload: dict[str, Any],
    ) -> AgentProposal:
        self.require_active_project(project_id)
        agent_task = self.get_agent_task(agent_task_id)
        if agent_task.project_id != project_id:
            raise ValueError(
                "Agent proposals must belong to the same project as their agent task: "
                f"{agent_task_id}"
            )
        if agent_task.role != role:
            raise ValueError(
                "Agent proposal role must match the source agent task role: "
                f"{agent_task.role} != {role}"
            )
        now = utc_now_iso()
        proposal = AgentProposal(
            proposal_id=new_proposal_id(),
            project_id=project_id,
            agent_task_id=agent_task_id,
            role=role,
            proposal_type=proposal_type,
            payload=dict(payload),
            created_at=now,
            updated_at=now,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_proposals (
                    proposal_id, project_id, agent_task_id, role, proposal_type,
                    payload_json, status, reason, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.proposal_id,
                    proposal.project_id,
                    proposal.agent_task_id,
                    proposal.role,
                    proposal.proposal_type,
                    _dumps(proposal.payload),
                    proposal.status,
                    proposal.reason,
                    proposal.created_at,
                    proposal.updated_at,
                ),
            )
        self.append_event(
            project_id=project_id,
            actor=role,
            action="agent_proposal_created",
            subject_type="agent_proposal",
            subject_id=proposal.proposal_id,
            payload=proposal.to_record(),
        )
        return proposal

    def list_agent_proposals(
        self,
        project_id: str,
        *,
        status: str | None = None,
    ) -> list[AgentProposal]:
        query = "SELECT * FROM agent_proposals WHERE project_id = ?"
        params: list[Any] = [project_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at ASC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._proposal_from_row(row) for row in rows]

    def get_agent_proposal(self, proposal_id: str) -> AgentProposal:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown proposal_id: {proposal_id}")
        return self._proposal_from_row(row)

    def update_agent_proposal_status(
        self,
        proposal_id: str,
        status: str,
        *,
        reason: str | None = None,
    ) -> AgentProposal:
        if status not in PROPOSAL_STATUSES:
            raise ValueError(f"Unsupported proposal status: {status}")
        proposal = self.get_agent_proposal(proposal_id)
        self._validate_agent_proposal_status_transition(proposal.status, status)
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE agent_proposals
                SET status = ?, reason = ?, updated_at = ?
                WHERE proposal_id = ?
                """,
                (status, reason, now, proposal_id),
            )
            row = conn.execute(
                "SELECT * FROM agent_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown proposal_id: {proposal_id}")
        return self._proposal_from_row(row)

    def append_event(
        self,
        *,
        project_id: str | None,
        actor: str,
        action: str,
        subject_type: str,
        subject_id: str,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        self.prepare()
        event = Event(
            event_id=new_event_id(),
            project_id=project_id,
            actor=actor,
            action=action,
            subject_type=subject_type,
            subject_id=subject_id,
            payload=dict(payload or {}),
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, project_id, actor, action, subject_type, subject_id,
                    payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.project_id,
                    event.actor,
                    event.action,
                    event.subject_type,
                    event.subject_id,
                    _dumps(event.payload),
                    event.created_at,
                ),
            )
        return event

    def list_events(
        self,
        *,
        project_id: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        subject_type: str | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        query = "SELECT * FROM events WHERE 1 = 1"
        params: list[Any] = []
        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)
        if actor is not None:
            query += " AND actor = ?"
            params.append(actor)
        if action is not None:
            query += " AND action = ?"
            params.append(action)
        if subject_type is not None:
            query += " AND subject_type = ?"
            params.append(subject_type)
        query += " ORDER BY created_at ASC, event_id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._event_from_row(row) for row in rows]

    def get_project_state(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        return {
            "project": project.to_record(),
            "hypotheses": [item.to_record() for item in self.list_hypotheses(project_id)],
            "tasks": [item.to_record() for item in self.list_research_tasks(project_id)],
            "candidate_operators": [
                item.to_record() for item in self.list_candidate_operators(project_id)
            ],
            "agent_tasks": [item.to_record() for item in self.list_agent_tasks(project_id)],
            "agent_proposals": [
                item.to_record() for item in self.list_agent_proposals(project_id)
            ],
            "events": [item.to_record() for item in self.list_events(project_id=project_id)],
        }

    def export_project_summary(self, project_id: str) -> dict[str, Any]:
        state = self.get_project_state(project_id)
        tasks = state["tasks"]
        completed = [task for task in tasks if task["status"] == "completed"]
        failed = [task for task in tasks if task["status"] == "failed"]
        task_status_counts = {status: 0 for status in sorted(TASK_STATUSES)}
        for task in tasks:
            task_status_counts[str(task["status"])] += 1
        agent_task_status_counts = {status: 0 for status in sorted(AGENT_TASK_STATUSES)}
        for task in state["agent_tasks"]:
            agent_task_status_counts[str(task["status"])] += 1
        experiment_run_budget = self.get_experiment_run_budget(project_id)
        scores = [
            task["result"].get("score")
            for task in completed
            if isinstance(task.get("result"), dict) and task["result"].get("score") is not None
        ]
        return {
            "project_id": project_id,
            "goal": state["project"]["goal"],
            "status": state["project"]["status"],
            "num_hypotheses": len(state["hypotheses"]),
            "num_tasks": len(tasks),
            "num_completed_tasks": len(completed),
            "num_failed_tasks": len(failed),
            "task_status_counts": task_status_counts,
            "agent_task_status_counts": agent_task_status_counts,
            "experiment_run_budget": experiment_run_budget,
            "project_concurrency": self.get_project_concurrency(project_id),
            "best_score": max(scores) if scores else None,
            "tasks": tasks,
        }

    def get_experiment_run_budget(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        limit = _experiment_run_limit(project)
        used = self._count_budgeted_experiment_runs(project_id)
        return {
            "limit": limit,
            "used": used,
            "remaining": None if limit is None else max(0, limit - used),
        }

    def get_project_concurrency(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        limit = _project_concurrency_limit(project)
        active = self.count_active_project_tasks(project_id)
        return {
            "limit": limit,
            "active": active,
            "remaining": None if limit is None else max(0, limit - active),
        }

    def count_active_project_tasks(self, project_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM research_tasks
                WHERE project_id = ?
                  AND status IN ('scheduled', 'running')
                """,
                (project_id,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def _validate_task_type(self, task_type: str) -> None:
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported research task type: {task_type}")

    def _validate_operator_refs(self, spec: dict[str, Any]) -> None:
        for ref in spec.get("operator_refs", []):
            if isinstance(ref, dict):
                ref_type = ref.get("type")
                operator_id = ref.get("id") or ref.get("candidate_operator_id")
            else:
                ref_type = "stable"
                operator_id = str(ref)
            if ref_type != "candidate":
                continue
            candidate = self.get_candidate_operator(str(operator_id))
            if candidate.status not in {"validated", "promoted"}:
                raise ValueError(
                    "Experiment specs can only reference validated or promoted "
                    f"candidate operators: {operator_id}"
                )

    def _validate_project_budget(
        self,
        project: ResearchProject,
        task_type: str,
        *,
        reserved_experiment_runs: int = 0,
    ) -> None:
        if task_type != "experiment_run":
            return
        limit = _experiment_run_limit(project)
        if limit is None:
            return
        used = self._count_budgeted_experiment_runs(project.project_id)
        if used + reserved_experiment_runs >= limit:
            raise ValueError(
                "Project experiment_run budget exhausted: "
                f"used {used} of {limit}, reserved {reserved_experiment_runs}"
            )

    def _count_budgeted_experiment_runs(self, project_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM research_tasks
                WHERE project_id = ?
                  AND task_type = 'experiment_run'
                  AND status != 'cancelled'
                """,
                (project_id,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def _validate_task_dependencies(self, project_id: str, dependencies: list[str]) -> None:
        for dependency_id in dependencies:
            dependency = self.get_research_task(dependency_id)
            if dependency.project_id != project_id:
                raise ValueError(
                    "Research task dependencies must belong to the same project: "
                    f"{dependency_id}"
                )

    def _validate_task_status_transition(self, current_status: str, next_status: str) -> None:
        if current_status == next_status:
            return
        if current_status in TERMINAL_TASK_STATUSES:
            raise ValueError(
                "Cannot transition terminal research task status through update_task_status: "
                f"{current_status} -> {next_status}. Use requeue_research_task when retrying "
                "failed or cancelled work."
            )

    def _validate_agent_task_status_transition(
        self,
        current_status: str,
        next_status: str,
    ) -> None:
        if current_status == next_status:
            return
        if current_status in TERMINAL_AGENT_TASK_STATUSES:
            raise ValueError(
                "Cannot transition terminal agent task status: "
                f"{current_status} -> {next_status}"
            )

    def _validate_agent_proposal_status_transition(
        self,
        current_status: str,
        next_status: str,
    ) -> None:
        if current_status == next_status:
            return
        if current_status in TERMINAL_PROPOSAL_STATUSES:
            raise ValueError(
                "Cannot transition terminal agent proposal status: "
                f"{current_status} -> {next_status}"
            )

    def _blocking_dependencies(self, task: ResearchTask) -> dict[str, str]:
        if not task.dependencies:
            return {}
        placeholders = _placeholders(task.dependencies)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT task_id, status
                FROM research_tasks
                WHERE task_id IN ({placeholders})
                """,
                task.dependencies,
            ).fetchall()
        statuses = {str(row["task_id"]): str(row["status"]) for row in rows}
        return {
            dependency_id: status
            for dependency_id, status in statuses.items()
            if status in {"failed", "cancelled", "blocked"}
        }

    def _cancel_open_project_work(self, project_id: str, *, actor: str) -> None:
        for task in self.list_research_tasks(project_id):
            if task.status in {"pending", "blocked", "scheduled", "running"}:
                self.cancel_research_task(
                    task.task_id,
                    actor=actor,
                    reason="Project cancelled.",
                )
        for agent_task in self.list_agent_tasks(project_id):
            if agent_task.status in {"pending", "running"}:
                self.update_agent_task_status(
                    agent_task.agent_task_id,
                    "cancelled",
                    error="Project cancelled.",
                )

    def _pause_scheduled_project_work(self, project_id: str, *, actor: str) -> None:
        for task in self.list_research_tasks(project_id, status="scheduled"):
            self.requeue_research_task(
                task.task_id,
                actor=actor,
                reason="Project paused.",
            )

    def _validate_project_can_complete(self, project_id: str) -> None:
        open_task_ids = [
            task.task_id
            for task in self.list_research_tasks(project_id)
            if task.status in {"pending", "blocked", "scheduled", "running"}
        ]
        open_agent_task_ids = [
            task.agent_task_id
            for task in self.list_agent_tasks(project_id)
            if task.status in {"pending", "running"}
        ]
        if open_task_ids or open_agent_task_ids:
            raise ValueError(
                "Cannot complete project with open work: "
                f"research_tasks={open_task_ids}, agent_tasks={open_agent_task_ids}"
            )

    def _validate_project_status_transition(self, current_status: str, next_status: str) -> None:
        if current_status == next_status:
            return
        if current_status in {"completed", "cancelled"}:
            raise ValueError(
                "Cannot transition terminal project status: "
                f"{current_status} -> {next_status}"
            )

    def _dependencies_completed(self, conn: sqlite3.Connection, task: ResearchTask) -> bool:
        if not task.dependencies:
            return True
        placeholders = ", ".join("?" for _ in task.dependencies)
        rows = conn.execute(
            f"SELECT status FROM research_tasks WHERE task_id IN ({placeholders})",
            task.dependencies,
        ).fetchall()
        return len(rows) == len(task.dependencies) and all(
            row["status"] == "completed" for row in rows
        )

    def _load_dependencies(self, conn: sqlite3.Connection, task_id: str) -> list[str]:
        rows = conn.execute(
            """
            SELECT depends_on_task_id
            FROM task_dependencies
            WHERE task_id = ?
            ORDER BY depends_on_task_id
            """,
            (task_id,),
        ).fetchall()
        return [row["depends_on_task_id"] for row in rows]

    def _project_from_row(self, row: sqlite3.Row) -> ResearchProject:
        return ResearchProject(
            project_id=row["project_id"],
            goal=row["goal"],
            dataset_path=row["dataset_path"],
            task_template=_loads(row["task_template_json"], {}),
            budget=_loads(row["budget_json"], {}),
            constraints=_loads(row["constraints_json"], {}),
            status=row["status"],
            summary=_loads(row["summary_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _hypothesis_from_row(self, row: sqlite3.Row) -> Hypothesis:
        return Hypothesis(
            hypothesis_id=row["hypothesis_id"],
            project_id=row["project_id"],
            title=row["title"],
            rationale=row["rationale"],
            status=row["status"],
            created_by=row["created_by"],
            metadata=_loads(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _research_task_from_row(
        self,
        row: sqlite3.Row,
        dependencies: list[str],
    ) -> ResearchTask:
        return ResearchTask(
            task_id=row["task_id"],
            project_id=row["project_id"],
            task_type=row["task_type"],
            spec=_loads(row["spec_json"], {}),
            status=row["status"],
            actor=row["actor"],
            priority=int(row["priority"]),
            dependencies=dependencies,
            resource_request=_loads(row["resource_request_json"], {}),
            result=_loads(row["result_json"], {}),
            error=row["error"],
            lease_id=row["lease_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def _candidate_from_row(self, row: sqlite3.Row) -> CandidateOperator:
        return CandidateOperator(
            candidate_operator_id=row["candidate_operator_id"],
            project_id=row["project_id"],
            hypothesis_id=row["hypothesis_id"],
            name=row["name"],
            spec=_loads(row["spec_json"], {}),
            status=row["status"],
            validation_report=_loads(row["validation_report_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _resource_from_row(self, row: sqlite3.Row) -> Resource:
        return Resource(
            resource_id=row["resource_id"],
            host=row["host"],
            kind=row["kind"],
            gpu_ids=_loads(row["gpu_ids_json"], []),
            max_concurrent_tasks=int(row["max_concurrent_tasks"]),
            tags=_loads(row["tags_json"], []),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _lease_from_row(self, row: sqlite3.Row) -> ResourceLease:
        return ResourceLease(
            lease_id=row["lease_id"],
            task_id=row["task_id"],
            resource_id=row["resource_id"],
            host=row["host"],
            gpu_ids=_loads(row["gpu_ids_json"], []),
            status=row["status"],
            created_at=row["created_at"],
            released_at=row["released_at"],
        )

    def _agent_task_from_row(self, row: sqlite3.Row) -> AgentTask:
        return AgentTask(
            agent_task_id=row["agent_task_id"],
            project_id=row["project_id"],
            role=row["role"],
            input_context=_loads(row["input_context_json"], {}),
            status=row["status"],
            output=_loads(row["output_json"], {}),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _proposal_from_row(self, row: sqlite3.Row) -> AgentProposal:
        return AgentProposal(
            proposal_id=row["proposal_id"],
            project_id=row["project_id"],
            agent_task_id=row["agent_task_id"],
            role=row["role"],
            proposal_type=row["proposal_type"],
            payload=_loads(row["payload_json"], {}),
            status=row["status"],
            reason=row["reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _event_from_row(self, row: sqlite3.Row) -> Event:
        return Event(
            event_id=row["event_id"],
            project_id=row["project_id"],
            actor=row["actor"],
            action=row["action"],
            subject_type=row["subject_type"],
            subject_id=row["subject_id"],
            payload=_loads(row["payload_json"], {}),
            created_at=row["created_at"],
        )


def _experiment_run_limit(project: ResearchProject) -> int | None:
    budget = dict(project.budget or {})
    template_budget = project.task_template.get("budget")
    if isinstance(template_budget, dict):
        budget = {**template_budget, **budget}
    raw_limit = budget.get("max_runs", budget.get("max_experiments"))
    if raw_limit is None:
        return None
    limit = int(raw_limit)
    return max(0, limit)


def _project_concurrency_limit(project: ResearchProject) -> int | None:
    merged: dict[str, Any] = {}
    template_budget = project.task_template.get("budget")
    if isinstance(template_budget, dict):
        merged.update(template_budget)
    template_constraints = project.task_template.get("constraints")
    if isinstance(template_constraints, dict):
        merged.update(template_constraints)
    merged.update(dict(project.budget or {}))
    merged.update(dict(project.constraints or {}))
    raw_limit = merged.get("max_concurrent_tasks", merged.get("max_concurrent_runs"))
    if raw_limit is None:
        return None
    limit = int(raw_limit)
    return max(0, limit)
