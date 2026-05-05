from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataos.llm.client import LLMClient
from dataos.research.models import AgentProposal, AgentTask
from dataos.research.store import ResearchStore


INITIAL_AGENT_ROLES = [
    "hypothesis",
    "experiment_planner",
    "operator_designer",
    "critic",
]
RESULT_AGENT_ROLES = [
    "evaluation_analyst",
    "critic",
    "operator_designer",
]


@dataclass(slots=True)
class ManagerDecision:
    action: str
    project_id: str
    agent_task_ids: list[str] = field(default_factory=list)
    proposal_ids: list[str] = field(default_factory=list)
    research_task_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "project_id": self.project_id,
            "agent_task_ids": self.agent_task_ids,
            "proposal_ids": self.proposal_ids,
            "research_task_ids": self.research_task_ids,
            "notes": self.notes,
        }


@dataclass(slots=True)
class AgentRunReport:
    project_id: str
    completed_agent_task_ids: list[str] = field(default_factory=list)
    proposal_ids: list[str] = field(default_factory=list)
    failed_agent_task_ids: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "completed_agent_task_ids": self.completed_agent_task_ids,
            "proposal_ids": self.proposal_ids,
            "failed_agent_task_ids": self.failed_agent_task_ids,
            "errors": self.errors,
        }


@dataclass(slots=True)
class AgentCycleReport:
    project_id: str
    first_decision: ManagerDecision
    agent_run: AgentRunReport
    final_decision: ManagerDecision

    def to_record(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "first_decision": self.first_decision.to_record(),
            "agent_run": self.agent_run.to_record(),
            "final_decision": self.final_decision.to_record(),
        }


@dataclass(slots=True)
class DataOSAgent:
    workspace_dir: Path | str
    llm_client: LLMClient | None = None
    store: ResearchStore = field(init=False)

    def __post_init__(self) -> None:
        self.store = ResearchStore(self.workspace_dir)
        self.store.prepare()

    def manager_tick(self, project_id: str) -> ManagerDecision:
        state = self.store.get_project_state(project_id)
        project_status = state["project"]["status"]
        if project_status not in {"created", "running"}:
            return ManagerDecision(
                action="project_inactive",
                project_id=project_id,
                notes=[f"Project status is {project_status}."],
            )
        proposed = self.store.list_agent_proposals(project_id, status="proposed")
        if proposed:
            accepted_task_ids: list[str] = []
            accepted_proposal_ids: list[str] = []
            notes: list[str] = []
            for proposal in proposed:
                try:
                    task_ids = self.accept_proposal(proposal.proposal_id)
                    accepted_task_ids.extend(task_ids)
                    accepted_proposal_ids.append(proposal.proposal_id)
                except Exception as exc:
                    self.reject_proposal(proposal.proposal_id, str(exc))
                    notes.append(f"Rejected {proposal.proposal_id}: {exc}")
            return ManagerDecision(
                action="accepted_proposals",
                project_id=project_id,
                proposal_ids=accepted_proposal_ids,
                research_task_ids=accepted_task_ids,
                notes=notes,
            )

        if (
            not state["agent_tasks"]
            and not state["hypotheses"]
            and not state["agent_proposals"]
        ):
            tasks = self.spawn_agent_tasks(project_id, INITIAL_AGENT_ROLES, "initial_campaign")
            return ManagerDecision(
                action="spawned_initial_agent_tasks",
                project_id=project_id,
                agent_task_ids=[task.agent_task_id for task in tasks],
            )

        if _has_new_completed_tasks_without_analysis(state):
            open_tasks = _open_agent_tasks_for_scope(
                state,
                "result_analysis",
                roles=RESULT_AGENT_ROLES,
            )
            if open_tasks:
                return ManagerDecision(
                    action="waiting_for_agent_tasks",
                    project_id=project_id,
                    agent_task_ids=[task["agent_task_id"] for task in open_tasks],
                    notes=["Result analysis already has open specialist tasks."],
                )
            tasks = self.spawn_agent_tasks(project_id, RESULT_AGENT_ROLES, "result_analysis")
            return ManagerDecision(
                action="spawned_result_agent_tasks",
                project_id=project_id,
                agent_task_ids=[task.agent_task_id for task in tasks],
            )

        if _campaign_ready_for_report(state):
            open_tasks = _open_agent_tasks_for_scope(state, "final_report", roles=["report"])
            if open_tasks:
                return ManagerDecision(
                    action="waiting_for_agent_tasks",
                    project_id=project_id,
                    agent_task_ids=[task["agent_task_id"] for task in open_tasks],
                    notes=["Final report already has an open specialist task."],
                )
            tasks = self.spawn_agent_tasks(project_id, ["report"], "final_report")
            return ManagerDecision(
                action="spawned_report_agent_task",
                project_id=project_id,
                agent_task_ids=[task.agent_task_id for task in tasks],
            )

        return ManagerDecision(action="no_op", project_id=project_id)

    def spawn_agent_tasks(
        self,
        project_id: str,
        roles: list[str],
        context_scope: str,
    ) -> list[AgentTask]:
        context = compile_context(self.store, project_id, context_scope=context_scope)
        return [
            self.store.create_agent_task(
                project_id,
                role=role,
                input_context={**context, "role": role, "context_scope": context_scope},
            )
            for role in roles
        ]

    def run_agent_task(self, agent_task_id: str) -> AgentProposal:
        task = self.store.get_agent_task(agent_task_id)
        _require_active_project(self.store, task.project_id)
        if task.status not in {"pending", "failed"}:
            raise ValueError(f"Agent task is not runnable: {agent_task_id}")
        self.store.update_agent_task_status(agent_task_id, "running")
        try:
            if self.llm_client is None:
                raise ValueError("llm_client is required to run specialist agent tasks")
            catalog = (
                task.input_context.get("operator_catalog")
                if isinstance(task.input_context, dict)
                else None
            )
            payload = self.llm_client.generate_json(
                _role_system_prompt(task.role, catalog),
                json.dumps(task.input_context, sort_keys=True, ensure_ascii=False),
                temperature=0.0,
            )
            proposal_type = _normalize_proposal_type(task.role, payload)
            proposal_payload = _normalize_proposal_payload(payload)
            _validate_proposal_payload(proposal_type, proposal_payload)
            output = {
                "proposal_type": proposal_type,
                "payload": proposal_payload,
            }
            self.store.update_agent_task_status(agent_task_id, "completed", output=output)
            return self.store.create_agent_proposal(
                project_id=task.project_id,
                agent_task_id=agent_task_id,
                role=task.role,
                proposal_type=proposal_type,
                payload=proposal_payload,
            )
        except Exception as exc:
            self.store.update_agent_task_status(agent_task_id, "failed", error=str(exc))
            raise

    def run_pending_agent_tasks(
        self,
        project_id: str,
        *,
        roles: list[str] | None = None,
        max_workers: int = 4,
        retry_failed: bool = False,
    ) -> AgentRunReport:
        _require_active_project(self.store, project_id)
        pending = self.store.list_agent_tasks(project_id, status="pending")
        if retry_failed:
            pending.extend(self.store.list_agent_tasks(project_id, status="failed"))
        if roles is not None:
            role_set = set(roles)
            pending = [task for task in pending if task.role in role_set]
        report = AgentRunReport(project_id=project_id)
        if not pending:
            return report
        worker_count = max(1, min(max_workers, len(pending)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self.run_agent_task, task.agent_task_id): task.agent_task_id
                for task in pending
            }
            for future in as_completed(futures):
                agent_task_id = futures[future]
                try:
                    proposal = future.result()
                    report.completed_agent_task_ids.append(agent_task_id)
                    report.proposal_ids.append(proposal.proposal_id)
                except Exception as exc:  # pragma: no cover - covered through state assertions
                    report.failed_agent_task_ids.append(agent_task_id)
                    report.errors[agent_task_id] = str(exc)
        return report

    def run_cycle(
        self,
        project_id: str,
        *,
        roles: list[str] | None = None,
        max_workers: int = 4,
        retry_failed: bool = False,
    ) -> AgentCycleReport:
        first_decision = self.manager_tick(project_id)
        if first_decision.action == "project_inactive":
            agent_run = AgentRunReport(project_id=project_id)
            final_decision = first_decision
        else:
            agent_run = self.run_pending_agent_tasks(
                project_id,
                roles=roles,
                max_workers=max_workers,
                retry_failed=retry_failed,
            )
            final_decision = self.manager_tick(project_id)
        return AgentCycleReport(
            project_id=project_id,
            first_decision=first_decision,
            agent_run=agent_run,
            final_decision=final_decision,
        )

    def accept_proposal(self, proposal_id: str) -> list[str]:
        return accept_agent_proposal(self.store, proposal_id)

    def reject_proposal(self, proposal_id: str, reason: str) -> None:
        reject_agent_proposal(self.store, proposal_id, reason)


def accept_agent_proposal(store: ResearchStore, proposal_id: str) -> list[str]:
    proposal = _find_proposal(store, proposal_id)
    if proposal.status != "proposed":
        raise ValueError(f"Proposal is not proposed: {proposal_id}")
    _require_active_project(store, proposal.project_id)
    task_ids = _commit_proposal(store, proposal)
    store.update_agent_proposal_status(proposal_id, "accepted")
    return task_ids


def reject_agent_proposal(store: ResearchStore, proposal_id: str, reason: str) -> None:
    proposal = _find_proposal(store, proposal_id)
    if proposal.status != "proposed":
        raise ValueError(f"Proposal is not proposed: {proposal_id}")
    store.update_agent_proposal_status(proposal_id, "rejected", reason=reason)


def compile_context(
    store: ResearchStore,
    project_id: str,
    *,
    context_scope: str,
) -> dict[str, Any]:
    from dataos.operators import build_default_registry  # lazy: keeps module importable if operators unavailable

    state = store.get_project_state(project_id)
    summary = store.export_project_summary(project_id)
    try:
        operator_catalog = _build_operator_catalog(build_default_registry().describe_all())
    except Exception:
        operator_catalog = []
    return {
        "context_scope": context_scope,
        "project": state["project"],
        "summary": summary,
        "hypotheses": state["hypotheses"],
        "tasks": _compact_tasks(state["tasks"]),
        "candidate_operators": state["candidate_operators"],
        "agent_proposals": [
            proposal
            for proposal in state["agent_proposals"]
            if proposal["status"] in {"accepted", "rejected"}
        ],
        "recent_events": state.get("events", [])[-20:],
        "operator_catalog": operator_catalog,
    }


def _commit_proposal(store: ResearchStore, proposal: AgentProposal) -> list[str]:
    payload = proposal.payload
    if proposal.proposal_type == "hypotheses":
        for item in payload.get("hypotheses", []):
            store.add_hypothesis(
                proposal.project_id,
                title=str(item["title"]),
                rationale=str(item.get("rationale", "")),
                created_by=proposal.role,
                metadata=dict(item.get("metadata", {})),
            )
        return []
    if proposal.proposal_type == "research_tasks":
        task_ids: list[str] = []
        task_items = list(payload.get("tasks", []))
        _validate_research_task_proposal(store, proposal.project_id, task_items)
        for item in task_items:
            task = store.submit_research_task(
                proposal.project_id,
                task_type=str(item["task_type"]),
                spec=dict(item.get("spec", {})),
                dependencies=list(item.get("dependencies", [])),
                actor=proposal.role,
                priority=int(item.get("priority", 0)),
                resource_request=dict(item.get("resource_request", {})),
            )
            task_ids.append(task.task_id)
        return task_ids
    if proposal.proposal_type == "candidate_operator":
        candidate = store.create_candidate_operator(
            proposal.project_id,
            name=str(payload["name"]),
            spec=dict(payload.get("spec", {})),
            hypothesis_id=payload.get("hypothesis_id"),
            actor=proposal.role,
        )
        validation = payload.get("validation_task")
        if validation is None:
            return []
        task = store.submit_research_task(
            proposal.project_id,
            task_type="operator_validation",
            spec={
                **dict(validation.get("spec", {})),
                "candidate_operator_id": candidate.candidate_operator_id,
            },
            actor=proposal.role,
            priority=int(validation.get("priority", 0)),
            resource_request=dict(validation.get("resource_request", {})),
        )
        return [task.task_id]
    if proposal.proposal_type == "report":
        store.update_project_summary(
            proposal.project_id,
            dict(payload.get("summary", payload)),
            status=str(payload.get("project_status", "completed")),
            actor=proposal.role,
        )
        return []
    if proposal.proposal_type in {"analysis", "critique"}:
        event_payload = dict(payload)
        if proposal.proposal_type == "analysis":
            covered_task_ids = event_payload.get("covered_task_ids")
            if covered_task_ids is None:
                covered_task_ids = [
                    task.task_id
                    for task in store.list_research_tasks(proposal.project_id, status="completed")
                ]
                event_payload["covered_task_ids"] = covered_task_ids
        store.append_event(
            project_id=proposal.project_id,
            actor=proposal.role,
            action=f"{proposal.proposal_type}_recorded",
            subject_type="agent_proposal",
            subject_id=proposal.proposal_id,
            payload=event_payload,
        )
        return []
    raise ValueError(f"Unsupported proposal_type: {proposal.proposal_type}")


def _find_proposal(store: ResearchStore, proposal_id: str) -> AgentProposal:
    return store.get_agent_proposal(proposal_id)


def _require_active_project(store: ResearchStore, project_id: str) -> None:
    project = store.get_project(project_id)
    if project.status not in {"created", "running"}:
        raise ValueError(f"Project is not active: {project_id} ({project.status})")


def _normalize_proposal_type(role: str, payload: dict[str, Any]) -> str:
    proposal_type = payload.get("proposal_type")
    if proposal_type:
        return str(proposal_type)
    defaults = {
        "hypothesis": "hypotheses",
        "experiment_planner": "research_tasks",
        "operator_designer": "candidate_operator",
        "evaluation_analyst": "analysis",
        "critic": "critique",
        "report": "report",
    }
    if role not in defaults:
        raise ValueError(f"Role must return proposal_type: {role}")
    return defaults[role]


def _normalize_proposal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "payload" in payload and isinstance(payload["payload"], dict):
        return dict(payload["payload"])
    return {key: value for key, value in payload.items() if key != "proposal_type"}


def _validate_proposal_payload(proposal_type: str, payload: dict[str, Any]) -> None:
    if proposal_type == "hypotheses":
        hypotheses = payload.get("hypotheses")
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError("hypotheses proposals require a non-empty hypotheses list")
        for item in hypotheses:
            if not isinstance(item, dict) or not item.get("title"):
                raise ValueError("Each hypothesis proposal item requires a title")
        return
    if proposal_type == "research_tasks":
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("research_tasks proposals require a non-empty tasks list")
        for item in tasks:
            if not isinstance(item, dict) or not item.get("task_type"):
                raise ValueError("Each research task proposal requires task_type")
            if "spec" in item and not isinstance(item["spec"], dict):
                raise ValueError("Research task proposal spec must be an object")
        return
    if proposal_type == "candidate_operator":
        if not payload.get("name") or not isinstance(payload.get("spec"), dict):
            raise ValueError("candidate_operator proposals require name and spec")
        validation_task = payload.get("validation_task")
        if validation_task is not None and not isinstance(validation_task, dict):
            raise ValueError("candidate_operator validation_task must be an object")
        return
    if proposal_type == "report":
        if "summary" in payload and not isinstance(payload["summary"], dict):
            raise ValueError("report summary must be an object")
        return
    if proposal_type in {"analysis", "critique"}:
        return
    raise ValueError(f"Unsupported proposal_type: {proposal_type}")


_OPERATOR_AWARE_ROLES: frozenset[str] = frozenset(
    {"experiment_planner", "operator_designer", "critic"}
)


def _role_system_prompt(
    role: str,
    operator_catalog: list[dict[str, str]] | None = None,
) -> str:
    base = (
        "You are a DataOS specialist agent. Return JSON only. "
        f"Your role is {role}. Produce a proposal_type and payload suitable for "
        "DataOS Research OS. Do not execute work directly."
    )
    if role not in _OPERATOR_AWARE_ROLES or not operator_catalog:
        return base
    lines = "\n".join(f"  {entry['summary']}" for entry in operator_catalog)
    return (
        f"{base}\n\n"
        "Available data operators (use exact names in experiment specs):\n"
        f"{lines}"
    )


def _build_operator_catalog(
    descriptions: list[dict[str, str]],
) -> list[dict[str, str]]:
    result = []
    for d in descriptions:
        name = str(d.get("name", ""))
        desc = str(d.get("description", ""))
        summary = f"{name} — {desc}"
        if len(summary) > 80:
            summary = summary[:77] + "..."
        result.append({"name": name, "summary": summary})
    return result


def _compact_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for task in tasks:
        compact.append(
            {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "status": task["status"],
                "priority": task["priority"],
                "dependencies": task["dependencies"],
                "result": task.get("result", {}),
                "error": task.get("error"),
            }
        )
    return compact


def _has_new_completed_tasks_without_analysis(state: dict[str, Any]) -> bool:
    completed = [task for task in state["tasks"] if task["status"] == "completed"]
    if not completed:
        return False
    covered_task_ids: set[str] = set()
    for event in state.get("events", []):
        if event.get("action") != "analysis_recorded":
            continue
        payload = event.get("payload", {})
        if isinstance(payload, dict):
            covered_task_ids.update(str(task_id) for task_id in payload.get("covered_task_ids", []))
    return any(task["task_id"] not in covered_task_ids for task in completed)


def _campaign_ready_for_report(state: dict[str, Any]) -> bool:
    completed = [task for task in state["tasks"] if task["status"] == "completed"]
    if not completed:
        return False
    reports = [
        proposal
        for proposal in state["agent_proposals"]
        if proposal["proposal_type"] == "report"
    ]
    if reports:
        return False
    open_research_tasks = [
        task
        for task in state["tasks"]
        if task["status"] in {"pending", "blocked", "scheduled", "running"}
    ]
    if open_research_tasks:
        return False
    return len(completed) >= 2 or _experiment_budget_exhausted(state)


def _experiment_budget_exhausted(state: dict[str, Any]) -> bool:
    project = state["project"]
    budget = dict(project.get("budget", {}))
    template_budget = project.get("task_template", {}).get("budget")
    if isinstance(template_budget, dict):
        budget = {**template_budget, **budget}
    raw_limit = budget.get("max_runs", budget.get("max_experiments"))
    if raw_limit is None:
        return False
    limit = max(0, int(raw_limit))
    used = sum(
        1
        for task in state["tasks"]
        if task["task_type"] == "experiment_run" and task["status"] != "cancelled"
    )
    return used >= limit


def _validate_research_task_proposal(
    store: ResearchStore,
    project_id: str,
    task_items: list[dict[str, Any]],
) -> None:
    reserved_experiment_runs = 0
    for item in task_items:
        task_type = str(item["task_type"])
        store.validate_research_task_request(
            project_id,
            task_type=task_type,
            spec=dict(item.get("spec", {})),
            dependencies=list(item.get("dependencies", [])),
            reserved_experiment_runs=reserved_experiment_runs,
        )
        if task_type == "experiment_run":
            reserved_experiment_runs += 1


def _open_agent_tasks_for_scope(
    state: dict[str, Any],
    context_scope: str,
    *,
    roles: list[str] | None = None,
) -> list[dict[str, Any]]:
    role_set = set(roles or [])
    open_statuses = {"pending", "running", "failed"}
    tasks = []
    for task in state["agent_tasks"]:
        if task["status"] not in open_statuses:
            continue
        if role_set and task["role"] not in role_set:
            continue
        input_context = task.get("input_context", {})
        if isinstance(input_context, dict) and input_context.get("context_scope") == context_scope:
            tasks.append(task)
    return tasks
