from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dataos.core.artifact import make_artifact_id, utc_now_iso


TaskType = Literal[
    "experiment_run",
    "eval_run",
    "operator_validation",
    "analysis_task",
    "artifact_check",
]
TaskStatus = Literal[
    "pending",
    "blocked",
    "scheduled",
    "running",
    "completed",
    "failed",
    "cancelled",
]
AgentTaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
ProposalStatus = Literal["proposed", "accepted", "rejected", "superseded"]
ProjectStatus = Literal["created", "running", "paused", "completed", "cancelled"]
CandidateOperatorStatus = Literal[
    "proposed",
    "generated",
    "validating",
    "validated",
    "failed",
    "promoted",
    "rejected",
]

TASK_TYPES = {
    "experiment_run",
    "eval_run",
    "operator_validation",
    "analysis_task",
    "artifact_check",
}
TASK_STATUSES = {
    "pending",
    "blocked",
    "scheduled",
    "running",
    "completed",
    "failed",
    "cancelled",
}
TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}
AGENT_TASK_STATUSES = {"pending", "running", "completed", "failed", "cancelled"}
TERMINAL_AGENT_TASK_STATUSES = {"completed", "cancelled"}
PROPOSAL_STATUSES = {"proposed", "accepted", "rejected", "superseded"}
TERMINAL_PROPOSAL_STATUSES = {"accepted", "rejected", "superseded"}
PROJECT_STATUSES = {"created", "running", "paused", "completed", "cancelled"}
ACTIVE_PROJECT_STATUSES = {"created", "running"}
CANDIDATE_OPERATOR_STATUSES = {
    "proposed",
    "generated",
    "validating",
    "validated",
    "failed",
    "promoted",
    "rejected",
}

AGENT_ROLES = {
    "research_manager",
    "hypothesis",
    "experiment_planner",
    "operator_designer",
    "evaluation_analyst",
    "critic",
    "report",
}


def new_project_id() -> str:
    return make_artifact_id("project")


def new_hypothesis_id() -> str:
    return make_artifact_id("hypothesis")


def new_task_id() -> str:
    return make_artifact_id("task")


def new_candidate_operator_id() -> str:
    return make_artifact_id("candidate-operator")


def new_agent_task_id() -> str:
    return make_artifact_id("agent-task")


def new_proposal_id() -> str:
    return make_artifact_id("proposal")


def new_event_id() -> str:
    return make_artifact_id("event")


def new_lease_id() -> str:
    return make_artifact_id("lease")


@dataclass(slots=True)
class ResearchProject:
    project_id: str
    goal: str
    dataset_path: str
    task_template: dict[str, Any]
    budget: dict[str, Any]
    constraints: dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    summary: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_record(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "goal": self.goal,
            "dataset_path": self.dataset_path,
            "task_template": self.task_template,
            "budget": self.budget,
            "constraints": self.constraints,
            "status": self.status,
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class Hypothesis:
    hypothesis_id: str
    project_id: str
    title: str
    rationale: str
    status: str = "active"
    created_by: str = "dataos_agent"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_record(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "project_id": self.project_id,
            "title": self.title,
            "rationale": self.rationale,
            "status": self.status,
            "created_by": self.created_by,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class ResearchTask:
    task_id: str
    project_id: str
    task_type: str
    spec: dict[str, Any]
    status: str = "pending"
    actor: str = "human"
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)
    resource_request: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    lease_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "task_type": self.task_type,
            "spec": self.spec,
            "status": self.status,
            "actor": self.actor,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "resource_request": self.resource_request,
            "result": self.result,
            "error": self.error,
            "lease_id": self.lease_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(slots=True)
class CandidateOperator:
    candidate_operator_id: str
    project_id: str
    name: str
    spec: dict[str, Any]
    status: str = "proposed"
    hypothesis_id: str | None = None
    validation_report: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_operator_id": self.candidate_operator_id,
            "project_id": self.project_id,
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "spec": self.spec,
            "status": self.status,
            "validation_report": self.validation_report,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class Resource:
    resource_id: str
    host: str
    kind: str = "local"
    gpu_ids: list[str] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_record(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "host": self.host,
            "kind": self.kind,
            "gpu_ids": self.gpu_ids,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "tags": self.tags,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class ResourceLease:
    lease_id: str
    task_id: str
    resource_id: str
    host: str
    gpu_ids: list[str] = field(default_factory=list)
    status: str = "active"
    created_at: str = field(default_factory=utc_now_iso)
    released_at: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "host": self.host,
            "gpu_ids": self.gpu_ids,
            "status": self.status,
            "created_at": self.created_at,
            "released_at": self.released_at,
        }


@dataclass(slots=True)
class AgentTask:
    agent_task_id: str
    project_id: str
    role: str
    input_context: dict[str, Any]
    status: str = "pending"
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_record(self) -> dict[str, Any]:
        return {
            "agent_task_id": self.agent_task_id,
            "project_id": self.project_id,
            "role": self.role,
            "input_context": self.input_context,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class AgentProposal:
    proposal_id: str
    project_id: str
    agent_task_id: str
    role: str
    proposal_type: str
    payload: dict[str, Any]
    status: str = "proposed"
    reason: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_record(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "project_id": self.project_id,
            "agent_task_id": self.agent_task_id,
            "role": self.role,
            "proposal_type": self.proposal_type,
            "payload": self.payload,
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class Event:
    event_id: str
    project_id: str | None
    actor: str
    action: str
    subject_type: str
    subject_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "project_id": self.project_id,
            "actor": self.actor,
            "action": self.action,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "payload": self.payload,
            "created_at": self.created_at,
        }
