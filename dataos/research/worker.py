from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataos.core.task import TaskTemplate
from dataos.llm.client import LLMClient
from dataos.research.store import ResearchStore
from dataos.runtime.runner import run_task


@dataclass(slots=True)
class WorkerReport:
    task_id: str
    status: str
    result: dict[str, Any]
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


def worker_run(
    task_id: str,
    workspace_dir: str | Path,
    *,
    llm_client: LLMClient | None = None,
) -> WorkerReport:
    store = ResearchStore(workspace_dir)
    store.prepare()
    task = store.get_research_task(task_id)
    if task.status not in {"running", "scheduled"}:
        raise ValueError(f"Task must be running or scheduled before worker execution: {task_id}")
    if task.status == "scheduled":
        store.require_active_project(task.project_id)
        task = store.update_task_status(task_id, "running", actor="worker")
    lease = store.get_lease(task.lease_id) if task.lease_id is not None else None

    try:
        result = _execute_task(store, task, llm_client=llm_client, lease=lease)
        completed = store.update_task_status(
            task_id,
            "completed",
            actor="worker",
            result=result,
            lease_id=task.lease_id,
        )
        if task.lease_id is not None:
            store.release_lease(task.lease_id)
        return WorkerReport(task_id=task_id, status=completed.status, result=result)
    except Exception as exc:
        failed = store.update_task_status(
            task_id,
            "failed",
            actor="worker",
            error=str(exc),
            lease_id=task.lease_id,
        )
        if task.lease_id is not None:
            store.release_lease(task.lease_id)
        return WorkerReport(
            task_id=task_id,
            status=failed.status,
            result=failed.result,
            error=failed.error,
        )


def _execute_task(
    store: ResearchStore,
    task,
    *,
    llm_client: LLMClient | None,
    lease,
) -> dict[str, Any]:
    if "stub_result" in task.spec:
        result = dict(task.spec["stub_result"])
        _apply_stub_side_effects(store, task, result)
        return result
    if lease is not None and lease.host != "local":
        return _run_ssh_command_task(task, lease)
    if task.task_type == "experiment_run":
        return _run_experiment_task(store, task, llm_client=llm_client)
    if task.task_type == "operator_validation":
        return _run_operator_validation_task(store, task)
    raise NotImplementedError(
        f"{task.task_type} worker requires a concrete implementation or stub_result"
    )


def _run_ssh_command_task(task, lease) -> dict[str, Any]:
    command = task.spec.get("ssh_command")
    if not command:
        raise NotImplementedError(
            "SSH-assigned tasks require spec.ssh_command in v1; "
            "use stub_result for tests or a local resource for run_task execution"
        )
    env_prefix = ""
    if lease.gpu_ids:
        env_prefix = f"CUDA_VISIBLE_DEVICES={','.join(lease.gpu_ids)} "
    completed = subprocess.run(
        ["ssh", lease.host, env_prefix + str(command)],
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "ssh_host": lease.host,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_experiment_task(
    store: ResearchStore,
    task,
    *,
    llm_client: LLMClient | None,
) -> dict[str, Any]:
    project = store.get_project(task.project_id)
    task_template = task.spec.get("task_template", project.task_template)
    dataset_path = task.spec.get("dataset_path", project.dataset_path)
    sandbox_root = task.spec.get("sandbox_root")
    result = run_task(
        task_template=TaskTemplate.from_dict(task_template),
        dataset_path=dataset_path,
        workspace_dir=store.workspace_dir,
        sandbox_root=sandbox_root,
        llm_client=llm_client,
        history_summaries=task.spec.get("history_summaries"),
    )
    artifact_records = [artifact.to_record() for artifact in result.experiment.artifacts]
    return {
        "experiment_id": result.experiment.exp_id,
        "status": result.experiment.status,
        "score": result.experiment.score,
        "sandbox": str(result.experiment.sandbox.root),
        "experiment_artifact_id": result.experiment_artifact.artifact_id,
        "archived_experiment_artifact_id": result.archived_experiment_artifact.artifact_id,
        "artifacts": artifact_records,
        "review": result.review,
    }


def _run_operator_validation_task(store: ResearchStore, task) -> dict[str, Any]:
    candidate_id = task.spec.get("candidate_operator_id")
    validation_report = dict(task.spec.get("validation_report", {}))
    if candidate_id is None:
        raise ValueError("operator_validation requires candidate_operator_id")
    status = "validated" if validation_report.get("ok", True) else "failed"
    store.update_candidate_operator_status(
        str(candidate_id),
        status,
        validation_report=validation_report,
    )
    return {
        "candidate_operator_id": candidate_id,
        "candidate_operator_status": status,
        "validation_report": validation_report,
    }


def _apply_stub_side_effects(store: ResearchStore, task, result: dict[str, Any]) -> None:
    if task.task_type != "operator_validation":
        return
    candidate_id = task.spec.get("candidate_operator_id")
    if candidate_id is None:
        return
    status = result.get("candidate_operator_status")
    if status in {"validated", "failed", "promoted", "rejected"}:
        store.update_candidate_operator_status(
            str(candidate_id),
            status,
            validation_report=dict(result.get("validation_report", result)),
        )
