from __future__ import annotations

from pathlib import Path
from typing import Any

from dataos.core.task import TaskTemplate
from dataos.llm.client import LLMClient
from dataos.research.agent import (
    AgentCycleReport,
    DataOSAgent,
    AgentRunReport,
    ManagerDecision,
    accept_agent_proposal,
    reject_agent_proposal,
)
from dataos.research.orchestrator import (
    ProjectStepReport,
    RunReadyReport,
    run_project_step,
    run_ready_tasks,
)
from dataos.research.scheduler import (
    RecoveryReport,
    SchedulerReport,
    recover_stale_tasks,
    scheduler_tick,
)
from dataos.research.store import ResearchStore
from dataos.research.worker import WorkerReport, worker_run


def create_project(
    *,
    goal: str,
    dataset_path: str | Path,
    task_template: TaskTemplate | dict[str, Any],
    budget: dict[str, Any],
    workspace_dir: str | Path,
    constraints: dict[str, Any] | None = None,
) -> str:
    store = ResearchStore(workspace_dir)
    project = store.create_project(
        goal=goal,
        dataset_path=dataset_path,
        task_template=task_template,
        budget=budget,
        constraints=constraints,
    )
    return project.project_id


def submit_research_task(
    *,
    project_id: str,
    task_type: str,
    spec: dict[str, Any],
    workspace_dir: str | Path,
    dependencies: list[str] | None = None,
    actor: str = "human",
    priority: int = 0,
    resource_request: dict[str, Any] | None = None,
) -> str:
    store = ResearchStore(workspace_dir)
    task = store.submit_research_task(
        project_id,
        task_type=task_type,
        spec=spec,
        dependencies=dependencies,
        actor=actor,
        priority=priority,
        resource_request=resource_request,
    )
    return task.task_id


def list_research_tasks(
    *,
    project_id: str,
    workspace_dir: str | Path,
    status: str | None = None,
    task_type: str | None = None,
) -> list[dict[str, Any]]:
    store = ResearchStore(workspace_dir)
    return [
        task.to_record()
        for task in store.list_research_tasks(project_id, status=status, task_type=task_type)
    ]


def get_research_task(task_id: str, workspace_dir: str | Path) -> dict[str, Any]:
    return ResearchStore(workspace_dir).get_research_task(task_id).to_record()


def cancel_research_task(
    task_id: str,
    workspace_dir: str | Path,
    *,
    actor: str = "human",
    reason: str | None = None,
) -> dict[str, Any]:
    return ResearchStore(workspace_dir).cancel_research_task(
        task_id,
        actor=actor,
        reason=reason,
    ).to_record()


def requeue_research_task(
    task_id: str,
    workspace_dir: str | Path,
    *,
    actor: str = "human",
    reason: str | None = None,
) -> dict[str, Any]:
    return ResearchStore(workspace_dir).requeue_research_task(
        task_id,
        actor=actor,
        reason=reason,
    ).to_record()


def list_agent_tasks(
    *,
    project_id: str,
    workspace_dir: str | Path,
    status: str | None = None,
    role: str | None = None,
) -> list[dict[str, Any]]:
    store = ResearchStore(workspace_dir)
    return [
        task.to_record()
        for task in store.list_agent_tasks(project_id, status=status, role=role)
    ]


def get_agent_task(agent_task_id: str, workspace_dir: str | Path) -> dict[str, Any]:
    return ResearchStore(workspace_dir).get_agent_task(agent_task_id).to_record()


def list_agent_proposals(
    *,
    project_id: str,
    workspace_dir: str | Path,
    status: str | None = None,
) -> list[dict[str, Any]]:
    store = ResearchStore(workspace_dir)
    return [
        proposal.to_record()
        for proposal in store.list_agent_proposals(project_id, status=status)
    ]


def get_agent_proposal(proposal_id: str, workspace_dir: str | Path) -> dict[str, Any]:
    return ResearchStore(workspace_dir).get_agent_proposal(proposal_id).to_record()


def list_resource_leases(
    *,
    workspace_dir: str | Path,
    status: str | None = None,
    resource_id: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    store = ResearchStore(workspace_dir)
    return [
        lease.to_record()
        for lease in store.list_leases(
            status=status,
            resource_id=resource_id,
            project_id=project_id,
        )
    ]


def list_candidate_operators(
    *,
    project_id: str,
    workspace_dir: str | Path,
    status: str | None = None,
) -> list[dict[str, Any]]:
    store = ResearchStore(workspace_dir)
    return [
        candidate.to_record()
        for candidate in store.list_candidate_operators(project_id, status=status)
    ]


def get_candidate_operator(candidate_operator_id: str, workspace_dir: str | Path) -> dict[str, Any]:
    return ResearchStore(workspace_dir).get_candidate_operator(candidate_operator_id).to_record()


def update_candidate_operator_status(
    *,
    candidate_operator_id: str,
    status: str,
    workspace_dir: str | Path,
    validation_report: dict[str, Any] | None = None,
    actor: str = "human",
) -> dict[str, Any]:
    store = ResearchStore(workspace_dir)
    return store.update_candidate_operator_status(
        candidate_operator_id,
        status,
        validation_report=validation_report,
        actor=actor,
    ).to_record()


def list_events(
    *,
    workspace_dir: str | Path,
    project_id: str | None = None,
    actor: str | None = None,
    action: str | None = None,
    subject_type: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    store = ResearchStore(workspace_dir)
    return [
        event.to_record()
        for event in store.list_events(
            project_id=project_id,
            actor=actor,
            action=action,
            subject_type=subject_type,
            limit=limit,
        )
    ]


def accept_proposal(
    *,
    proposal_id: str,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    store = ResearchStore(workspace_dir)
    task_ids = accept_agent_proposal(store, proposal_id)
    proposal = store.get_agent_proposal(proposal_id)
    return {
        "proposal": proposal.to_record(),
        "research_task_ids": task_ids,
    }


def reject_proposal(
    *,
    proposal_id: str,
    reason: str,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    store = ResearchStore(workspace_dir)
    reject_agent_proposal(store, proposal_id, reason)
    return {"proposal": store.get_agent_proposal(proposal_id).to_record()}


def get_project_state(project_id: str, workspace_dir: str | Path) -> dict[str, Any]:
    return ResearchStore(workspace_dir).get_project_state(project_id)


def compare_runs(project_id: str, workspace_dir: str | Path) -> dict[str, Any]:
    summary = ResearchStore(workspace_dir).export_project_summary(project_id)
    completed = [
        task for task in summary["tasks"]
        if task["status"] == "completed" and task.get("result", {}).get("score") is not None
    ]
    completed.sort(key=lambda item: item["result"]["score"], reverse=True)
    return {
        "project_id": project_id,
        "best_run": completed[0] if completed else None,
        "ranked_runs": completed,
        "summary": summary,
    }


def export_project_summary(project_id: str, workspace_dir: str | Path) -> dict[str, Any]:
    return ResearchStore(workspace_dir).export_project_summary(project_id)


def update_project_status(
    project_id: str,
    status: str,
    workspace_dir: str | Path,
    *,
    actor: str = "human",
) -> dict[str, Any]:
    return ResearchStore(workspace_dir).update_project_status(
        project_id,
        status,
        actor=actor,
    ).to_record()


def run_scheduler_tick(
    workspace_dir: str | Path,
    max_tasks: int | None = None,
    *,
    project_id: str | None = None,
) -> SchedulerReport:
    return scheduler_tick(workspace_dir, max_tasks=max_tasks, project_id=project_id)


def recover_stale_research_tasks(
    workspace_dir: str | Path,
    *,
    older_than_seconds: int,
    project_id: str | None = None,
    statuses: list[str] | None = None,
) -> RecoveryReport:
    return recover_stale_tasks(
        workspace_dir,
        older_than_seconds=older_than_seconds,
        project_id=project_id,
        statuses=statuses,
    )


def run_worker(task_id: str, workspace_dir: str | Path) -> WorkerReport:
    return worker_run(task_id, workspace_dir)


def run_ready_research_tasks(
    workspace_dir: str | Path,
    *,
    project_id: str | None = None,
    max_tasks: int | None = None,
    max_workers: int = 4,
) -> RunReadyReport:
    return run_ready_tasks(
        workspace_dir,
        project_id=project_id,
        max_tasks=max_tasks,
        max_workers=max_workers,
    )


def run_project_research_step(
    workspace_dir: str | Path,
    project_id: str,
    *,
    llm_client: LLMClient,
    agent_max_workers: int = 4,
    task_max_workers: int = 4,
    max_tasks: int | None = None,
    retry_failed_agent_tasks: bool = False,
) -> ProjectStepReport:
    return run_project_step(
        workspace_dir,
        project_id,
        llm_client=llm_client,
        agent_max_workers=agent_max_workers,
        task_max_workers=task_max_workers,
        max_tasks=max_tasks,
        retry_failed_agent_tasks=retry_failed_agent_tasks,
    )


def manager_tick(agent: DataOSAgent, project_id: str) -> ManagerDecision:
    return agent.manager_tick(project_id)


def run_pending_agent_tasks(
    agent: DataOSAgent,
    project_id: str,
    *,
    roles: list[str] | None = None,
    max_workers: int = 4,
    retry_failed: bool = False,
) -> AgentRunReport:
    return agent.run_pending_agent_tasks(
        project_id,
        roles=roles,
        max_workers=max_workers,
        retry_failed=retry_failed,
    )


def run_agent_cycle(
    agent: DataOSAgent,
    project_id: str,
    *,
    roles: list[str] | None = None,
    max_workers: int = 4,
    retry_failed: bool = False,
) -> AgentCycleReport:
    return agent.run_cycle(
        project_id,
        roles=roles,
        max_workers=max_workers,
        retry_failed=retry_failed,
    )
