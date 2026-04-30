from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataos.llm.client import LLMClient
from dataos.research.agent import AgentCycleReport, DataOSAgent
from dataos.research.scheduler import SchedulerReport, scheduler_tick
from dataos.research.worker import WorkerReport, worker_run


@dataclass(slots=True)
class RunReadyReport:
    scheduler: SchedulerReport
    worker_reports: list[WorkerReport] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler.to_record(),
            "worker_reports": [report.to_record() for report in self.worker_reports],
            "errors": self.errors,
        }


@dataclass(slots=True)
class ProjectStepReport:
    project_id: str
    agent_cycle: AgentCycleReport | None = None
    task_run: RunReadyReport | None = None
    post_agent_task_run: RunReadyReport | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "agent_cycle": self.agent_cycle.to_record() if self.agent_cycle else None,
            "task_run": self.task_run.to_record() if self.task_run else None,
            "post_agent_task_run": (
                self.post_agent_task_run.to_record() if self.post_agent_task_run else None
            ),
            "errors": self.errors,
        }


def run_ready_tasks(
    workspace_dir: str | Path,
    *,
    project_id: str | None = None,
    max_tasks: int | None = None,
    max_workers: int = 4,
    llm_client: LLMClient | None = None,
) -> RunReadyReport:
    scheduler_report = scheduler_tick(workspace_dir, max_tasks=max_tasks, project_id=project_id)
    report = RunReadyReport(scheduler=scheduler_report)
    task_ids = list(scheduler_report.scheduled_task_ids)
    if not task_ids:
        return report
    worker_count = max(1, min(max_workers, len(task_ids)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                worker_run,
                task_id,
                workspace_dir,
                llm_client=llm_client,
            ): task_id
            for task_id in task_ids
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                report.worker_reports.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive isolation
                report.errors[task_id] = str(exc)
    return report


def run_project_step(
    workspace_dir: str | Path,
    project_id: str,
    *,
    llm_client: LLMClient,
    agent_max_workers: int = 4,
    task_max_workers: int = 4,
    max_tasks: int | None = None,
    retry_failed_agent_tasks: bool = False,
) -> ProjectStepReport:
    report = ProjectStepReport(project_id=project_id)
    agent = DataOSAgent(workspace_dir, llm_client)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                agent.run_cycle,
                project_id,
                max_workers=agent_max_workers,
                retry_failed=retry_failed_agent_tasks,
            ): "agent_cycle",
            executor.submit(
                run_ready_tasks,
                workspace_dir,
                project_id=project_id,
                max_tasks=max_tasks,
                max_workers=task_max_workers,
                llm_client=llm_client,
            ): "task_run",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                setattr(report, key, future.result())
            except Exception as exc:  # pragma: no cover - defensive step isolation
                report.errors[key] = str(exc)
    if (
        report.agent_cycle is not None
        and report.agent_cycle.final_decision.research_task_ids
        and "post_agent_task_run" not in report.errors
    ):
        try:
            report.post_agent_task_run = run_ready_tasks(
                workspace_dir,
                project_id=project_id,
                max_tasks=max_tasks,
                max_workers=task_max_workers,
                llm_client=llm_client,
            )
        except Exception as exc:  # pragma: no cover - defensive step isolation
            report.errors["post_agent_task_run"] = str(exc)
    return report
