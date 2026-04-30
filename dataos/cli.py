from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dataos.core.task import TaskTemplate
from dataos.llm.openai_compat import (
    DASHSCOPE_BASE_URL,
    DEFAULT_MODEL,
    OpenAICompatLLMClient,
)
from dataos.research.agent import DataOSAgent
from dataos.research.api import (
    accept_proposal,
    cancel_research_task,
    compare_runs,
    create_project,
    export_project_summary,
    get_agent_proposal,
    get_agent_task,
    get_candidate_operator,
    get_project_state,
    get_research_task,
    list_events,
    list_agent_proposals,
    list_agent_tasks,
    list_candidate_operators,
    list_resource_leases,
    list_research_tasks,
    recover_stale_research_tasks,
    requeue_research_task,
    reject_proposal,
    run_agent_cycle,
    run_pending_agent_tasks,
    run_project_research_step,
    run_ready_research_tasks,
    run_scheduler_tick,
    run_worker,
    submit_research_task,
    update_project_status,
    update_candidate_operator_status,
)
from dataos.research.models import CANDIDATE_OPERATOR_STATUSES, PROJECT_STATUSES
from dataos.research.store import ResearchStore


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataos")
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers(dest="command")
    _add_project_commands(subparsers)
    _add_task_commands(subparsers)
    _add_resource_commands(subparsers)
    _add_candidate_commands(subparsers)
    _add_scheduler_commands(subparsers)
    _add_worker_commands(subparsers)
    _add_agent_commands(subparsers)
    _add_event_commands(subparsers)
    return parser


def _add_project_commands(subparsers) -> None:
    project = subparsers.add_parser("project")
    project_sub = project.add_subparsers(dest="project_command")

    create = project_sub.add_parser("create")
    _add_workspace_arg(create)
    create.add_argument("--goal", required=True)
    create.add_argument("--dataset-path", required=True)
    create.add_argument("--task-template", required=True)
    create.add_argument("--budget", required=True)
    create.add_argument("--constraints", default="{}")
    create.set_defaults(func=_cmd_project_create)

    state = project_sub.add_parser("state")
    _add_workspace_arg(state)
    state.add_argument("project_id")
    state.set_defaults(func=_cmd_project_state)

    summary = project_sub.add_parser("summary")
    _add_workspace_arg(summary)
    summary.add_argument("project_id")
    summary.set_defaults(func=_cmd_project_summary)

    compare = project_sub.add_parser("compare")
    _add_workspace_arg(compare)
    compare.add_argument("project_id")
    compare.set_defaults(func=_cmd_project_compare)

    step = project_sub.add_parser("step")
    _add_workspace_arg(step)
    step.add_argument("project_id")
    step.add_argument("--agent-workers", type=int, default=4)
    step.add_argument("--task-workers", type=int, default=4)
    step.add_argument("--max-tasks", type=int)
    step.add_argument("--retry-failed-agent-tasks", action="store_true")
    step.set_defaults(func=_cmd_project_step)

    set_status = project_sub.add_parser("set-status")
    _add_workspace_arg(set_status)
    set_status.add_argument("project_id")
    set_status.add_argument("--status", required=True, choices=sorted(PROJECT_STATUSES))
    set_status.add_argument("--actor", default="human")
    set_status.set_defaults(func=_cmd_project_set_status)


def _add_task_commands(subparsers) -> None:
    task = subparsers.add_parser("task")
    task_sub = task.add_subparsers(dest="task_command")

    submit = task_sub.add_parser("submit")
    _add_workspace_arg(submit)
    submit.add_argument("--project-id", required=True)
    submit.add_argument("--type", required=True, dest="task_type")
    submit.add_argument("--spec", required=True)
    submit.add_argument("--dependency", action="append", default=[])
    submit.add_argument("--actor", default="human")
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--resource-request", default="{}")
    submit.set_defaults(func=_cmd_task_submit)

    list_cmd = task_sub.add_parser("list")
    _add_workspace_arg(list_cmd)
    list_cmd.add_argument("--project-id", required=True)
    list_cmd.add_argument("--status")
    list_cmd.add_argument("--type", dest="task_type")
    list_cmd.set_defaults(func=_cmd_task_list)

    get_cmd = task_sub.add_parser("get")
    _add_workspace_arg(get_cmd)
    get_cmd.add_argument("task_id")
    get_cmd.set_defaults(func=_cmd_task_get)

    cancel = task_sub.add_parser("cancel")
    _add_workspace_arg(cancel)
    cancel.add_argument("task_id")
    cancel.add_argument("--reason")
    cancel.add_argument("--actor", default="human")
    cancel.set_defaults(func=_cmd_task_cancel)

    requeue = task_sub.add_parser("requeue")
    _add_workspace_arg(requeue)
    requeue.add_argument("task_id")
    requeue.add_argument("--reason")
    requeue.add_argument("--actor", default="human")
    requeue.set_defaults(func=_cmd_task_requeue)


def _add_resource_commands(subparsers) -> None:
    resource = subparsers.add_parser("resource")
    resource_sub = resource.add_subparsers(dest="resource_command")

    add = resource_sub.add_parser("add")
    _add_workspace_arg(add)
    add.add_argument("--resource-id", required=True)
    add.add_argument("--host", required=True)
    add.add_argument("--kind", choices=["local", "ssh"], default="local")
    add.add_argument("--gpu-ids", default="")
    add.add_argument("--max-concurrent-tasks", type=int, default=1)
    add.add_argument("--tags", default="")
    add.set_defaults(func=_cmd_resource_add)

    list_cmd = resource_sub.add_parser("list")
    _add_workspace_arg(list_cmd)
    list_cmd.set_defaults(func=_cmd_resource_list)

    leases = resource_sub.add_parser("leases")
    _add_workspace_arg(leases)
    leases.add_argument("--status")
    leases.add_argument("--resource-id")
    leases.add_argument("--project-id")
    leases.set_defaults(func=_cmd_resource_leases)


def _add_candidate_commands(subparsers) -> None:
    candidate = subparsers.add_parser("candidate")
    candidate_sub = candidate.add_subparsers(dest="candidate_command")

    list_cmd = candidate_sub.add_parser("list")
    _add_workspace_arg(list_cmd)
    list_cmd.add_argument("--project-id", required=True)
    list_cmd.add_argument("--status", choices=sorted(CANDIDATE_OPERATOR_STATUSES))
    list_cmd.set_defaults(func=_cmd_candidate_list)

    get_cmd = candidate_sub.add_parser("get")
    _add_workspace_arg(get_cmd)
    get_cmd.add_argument("candidate_operator_id")
    get_cmd.set_defaults(func=_cmd_candidate_get)

    set_status = candidate_sub.add_parser("set-status")
    _add_workspace_arg(set_status)
    set_status.add_argument("candidate_operator_id")
    set_status.add_argument("--status", required=True, choices=sorted(CANDIDATE_OPERATOR_STATUSES))
    set_status.add_argument("--validation-report", default="{}")
    set_status.add_argument("--actor", default="human")
    set_status.set_defaults(func=_cmd_candidate_set_status)


def _add_scheduler_commands(subparsers) -> None:
    scheduler = subparsers.add_parser("scheduler")
    scheduler_sub = scheduler.add_subparsers(dest="scheduler_command")

    tick = scheduler_sub.add_parser("tick")
    _add_workspace_arg(tick)
    tick.add_argument("--project-id")
    tick.add_argument("--max-tasks", type=int)
    tick.set_defaults(func=_cmd_scheduler_tick)

    run_ready = scheduler_sub.add_parser("run-ready")
    _add_workspace_arg(run_ready)
    run_ready.add_argument("--project-id")
    run_ready.add_argument("--max-tasks", type=int)
    run_ready.add_argument("--max-workers", type=int, default=4)
    run_ready.set_defaults(func=_cmd_scheduler_run_ready)

    recover = scheduler_sub.add_parser("recover-stale")
    _add_workspace_arg(recover)
    recover.add_argument("--project-id")
    recover.add_argument("--older-than-seconds", type=int, required=True)
    recover.add_argument("--status", action="append", choices=["scheduled", "running"])
    recover.set_defaults(func=_cmd_scheduler_recover_stale)


def _add_worker_commands(subparsers) -> None:
    worker = subparsers.add_parser("worker")
    worker_sub = worker.add_subparsers(dest="worker_command")

    run = worker_sub.add_parser("run")
    _add_workspace_arg(run)
    run.add_argument("task_id")
    run.set_defaults(func=_cmd_worker_run)


def _add_agent_commands(subparsers) -> None:
    agent = subparsers.add_parser("agent")
    agent_sub = agent.add_subparsers(dest="agent_command")

    manager = agent_sub.add_parser("manager-tick")
    _add_workspace_arg(manager)
    manager.add_argument("project_id")
    manager.set_defaults(func=_cmd_agent_manager_tick)

    run = agent_sub.add_parser("run-task")
    _add_workspace_arg(run)
    run.add_argument("agent_task_id")
    run.set_defaults(func=_cmd_agent_run_task)

    get_task = agent_sub.add_parser("task-get")
    _add_workspace_arg(get_task)
    get_task.add_argument("agent_task_id")
    get_task.set_defaults(func=_cmd_agent_task_get)

    run_pending = agent_sub.add_parser("run-pending")
    _add_workspace_arg(run_pending)
    run_pending.add_argument("--project-id", required=True)
    run_pending.add_argument("--role", action="append")
    run_pending.add_argument("--max-workers", type=int, default=4)
    run_pending.add_argument("--retry-failed", action="store_true")
    run_pending.set_defaults(func=_cmd_agent_run_pending)

    cycle = agent_sub.add_parser("cycle")
    _add_workspace_arg(cycle)
    cycle.add_argument("--project-id", required=True)
    cycle.add_argument("--role", action="append")
    cycle.add_argument("--max-workers", type=int, default=4)
    cycle.add_argument("--retry-failed", action="store_true")
    cycle.set_defaults(func=_cmd_agent_cycle)

    list_tasks = agent_sub.add_parser("task-list")
    _add_workspace_arg(list_tasks)
    list_tasks.add_argument("--project-id", required=True)
    list_tasks.add_argument("--status")
    list_tasks.add_argument("--role")
    list_tasks.set_defaults(func=_cmd_agent_task_list)

    list_proposals = agent_sub.add_parser("proposal-list")
    _add_workspace_arg(list_proposals)
    list_proposals.add_argument("--project-id", required=True)
    list_proposals.add_argument("--status")
    list_proposals.set_defaults(func=_cmd_agent_proposal_list)

    get_proposal = agent_sub.add_parser("proposal-get")
    _add_workspace_arg(get_proposal)
    get_proposal.add_argument("proposal_id")
    get_proposal.set_defaults(func=_cmd_agent_proposal_get)

    accept = agent_sub.add_parser("proposal-accept")
    _add_workspace_arg(accept)
    accept.add_argument("proposal_id")
    accept.set_defaults(func=_cmd_agent_proposal_accept)

    reject = agent_sub.add_parser("proposal-reject")
    _add_workspace_arg(reject)
    reject.add_argument("proposal_id")
    reject.add_argument("--reason", required=True)
    reject.set_defaults(func=_cmd_agent_proposal_reject)

    daemon = agent_sub.add_parser("daemon")
    daemon_sub = daemon.add_subparsers(dest="agent_daemon_command")
    daemon_run = daemon_sub.add_parser("run")
    _add_workspace_arg(daemon_run)
    daemon_run.add_argument("--project-id", required=True)
    daemon_run.add_argument("--generator-cadence", type=float, default=30.0)
    daemon_run.add_argument("--executor-cadence", type=float, default=2.0)
    daemon_run.add_argument("--analyzer-cadence", type=float, default=15.0)
    daemon_run.add_argument("--reporter-cadence", type=float, default=60.0)
    daemon_run.add_argument("--stale-seconds", type=int, default=900)
    daemon_run.add_argument("--max-specialist-workers", type=int, default=4)
    daemon_run.set_defaults(func=_cmd_agent_daemon_run)


def _add_event_commands(subparsers) -> None:
    event = subparsers.add_parser("event")
    event_sub = event.add_subparsers(dest="event_command")

    list_cmd = event_sub.add_parser("list")
    _add_workspace_arg(list_cmd)
    list_cmd.add_argument("--project-id")
    list_cmd.add_argument("--actor")
    list_cmd.add_argument("--action")
    list_cmd.add_argument("--subject-type")
    list_cmd.add_argument("--limit", type=int)
    list_cmd.set_defaults(func=_cmd_event_list)


def _add_workspace_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default="workspace")


def _cmd_project_create(args) -> dict[str, Any]:
    project_id = create_project(
        goal=args.goal,
        dataset_path=args.dataset_path,
        task_template=TaskTemplate.from_json(args.task_template),
        budget=_load_json_arg(args.budget),
        constraints=_load_json_arg(args.constraints),
        workspace_dir=args.workspace,
    )
    return {"project_id": project_id}


def _cmd_project_state(args) -> dict[str, Any]:
    return get_project_state(args.project_id, args.workspace)


def _cmd_project_summary(args) -> dict[str, Any]:
    return export_project_summary(args.project_id, args.workspace)


def _cmd_project_compare(args) -> dict[str, Any]:
    return compare_runs(args.project_id, args.workspace)


def _cmd_project_set_status(args) -> dict[str, Any]:
    return update_project_status(
        args.project_id,
        args.status,
        args.workspace,
        actor=args.actor,
    )


def _cmd_project_step(args) -> dict[str, Any]:
    return run_project_research_step(
        args.workspace,
        args.project_id,
        llm_client=_build_llm_client_from_env(),
        agent_max_workers=args.agent_workers,
        task_max_workers=args.task_workers,
        max_tasks=args.max_tasks,
        retry_failed_agent_tasks=args.retry_failed_agent_tasks,
    ).to_record()


def _cmd_task_submit(args) -> dict[str, Any]:
    task_id = submit_research_task(
        project_id=args.project_id,
        task_type=args.task_type,
        spec=_load_json_arg(args.spec),
        dependencies=list(args.dependency),
        actor=args.actor,
        priority=args.priority,
        resource_request=_load_json_arg(args.resource_request),
        workspace_dir=args.workspace,
    )
    return {"task_id": task_id}


def _cmd_task_list(args) -> list[dict[str, Any]]:
    return list_research_tasks(
        project_id=args.project_id,
        status=args.status,
        task_type=args.task_type,
        workspace_dir=args.workspace,
    )


def _cmd_task_get(args) -> dict[str, Any]:
    return get_research_task(args.task_id, args.workspace)


def _cmd_task_cancel(args) -> dict[str, Any]:
    return cancel_research_task(
        args.task_id,
        args.workspace,
        actor=args.actor,
        reason=args.reason,
    )


def _cmd_task_requeue(args) -> dict[str, Any]:
    return requeue_research_task(
        args.task_id,
        args.workspace,
        actor=args.actor,
        reason=args.reason,
    )


def _cmd_resource_add(args) -> dict[str, Any]:
    store = ResearchStore(args.workspace)
    resource = store.add_resource(
        resource_id=args.resource_id,
        host=args.host,
        kind=args.kind,
        gpu_ids=_split_csv(args.gpu_ids),
        max_concurrent_tasks=args.max_concurrent_tasks,
        tags=_split_csv(args.tags),
    )
    return resource.to_record()


def _cmd_resource_list(args) -> list[dict[str, Any]]:
    return [resource.to_record() for resource in ResearchStore(args.workspace).list_resources()]


def _cmd_resource_leases(args) -> list[dict[str, Any]]:
    return list_resource_leases(
        workspace_dir=args.workspace,
        status=args.status,
        resource_id=args.resource_id,
        project_id=args.project_id,
    )


def _cmd_candidate_list(args) -> list[dict[str, Any]]:
    return list_candidate_operators(
        project_id=args.project_id,
        workspace_dir=args.workspace,
        status=args.status,
    )


def _cmd_candidate_get(args) -> dict[str, Any]:
    return get_candidate_operator(args.candidate_operator_id, args.workspace)


def _cmd_candidate_set_status(args) -> dict[str, Any]:
    return update_candidate_operator_status(
        candidate_operator_id=args.candidate_operator_id,
        status=args.status,
        workspace_dir=args.workspace,
        validation_report=_load_json_arg(args.validation_report),
        actor=args.actor,
    )


def _cmd_scheduler_tick(args) -> dict[str, Any]:
    return run_scheduler_tick(
        args.workspace,
        max_tasks=args.max_tasks,
        project_id=args.project_id,
    ).to_record()


def _cmd_scheduler_run_ready(args) -> dict[str, Any]:
    return run_ready_research_tasks(
        args.workspace,
        project_id=args.project_id,
        max_tasks=args.max_tasks,
        max_workers=args.max_workers,
    ).to_record()


def _cmd_scheduler_recover_stale(args) -> dict[str, Any]:
    return recover_stale_research_tasks(
        args.workspace,
        older_than_seconds=args.older_than_seconds,
        project_id=args.project_id,
        statuses=args.status,
    ).to_record()


def _cmd_worker_run(args) -> dict[str, Any]:
    return run_worker(args.task_id, args.workspace).to_record()


def _cmd_agent_manager_tick(args) -> dict[str, Any]:
    agent = DataOSAgent(args.workspace)
    return agent.manager_tick(args.project_id).to_record()


def _cmd_agent_run_task(args) -> dict[str, Any]:
    agent = DataOSAgent(args.workspace, _build_llm_client_from_env())
    return agent.run_agent_task(args.agent_task_id).to_record()


def _cmd_agent_task_get(args) -> dict[str, Any]:
    return get_agent_task(args.agent_task_id, args.workspace)


def _cmd_agent_run_pending(args) -> dict[str, Any]:
    agent = DataOSAgent(args.workspace, _build_llm_client_from_env())
    return run_pending_agent_tasks(
        agent,
        args.project_id,
        roles=args.role,
        max_workers=args.max_workers,
        retry_failed=args.retry_failed,
    ).to_record()


def _cmd_agent_cycle(args) -> dict[str, Any]:
    agent = DataOSAgent(args.workspace, _build_llm_client_from_env())
    return run_agent_cycle(
        agent,
        args.project_id,
        roles=args.role,
        max_workers=args.max_workers,
        retry_failed=args.retry_failed,
    ).to_record()


def _cmd_agent_task_list(args) -> list[dict[str, Any]]:
    return list_agent_tasks(
        project_id=args.project_id,
        workspace_dir=args.workspace,
        status=args.status,
        role=args.role,
    )


def _cmd_agent_proposal_list(args) -> list[dict[str, Any]]:
    return list_agent_proposals(
        project_id=args.project_id,
        workspace_dir=args.workspace,
        status=args.status,
    )


def _cmd_agent_proposal_get(args) -> dict[str, Any]:
    return get_agent_proposal(args.proposal_id, args.workspace)


def _cmd_agent_proposal_accept(args) -> dict[str, Any]:
    return accept_proposal(proposal_id=args.proposal_id, workspace_dir=args.workspace)


def _cmd_agent_proposal_reject(args) -> dict[str, Any]:
    return reject_proposal(
        proposal_id=args.proposal_id,
        reason=args.reason,
        workspace_dir=args.workspace,
    )


def _cmd_agent_daemon_run(args) -> None:
    from dataos.agent import Daemon, DaemonCadences

    daemon = Daemon(
        workspace_dir=args.workspace,
        project_id=args.project_id,
        llm_client=_build_llm_client_from_env(),
        cadences=DaemonCadences(
            generator=args.generator_cadence,
            executor=args.executor_cadence,
            analyzer=args.analyzer_cadence,
            reporter=args.reporter_cadence,
        ),
        stale_seconds=args.stale_seconds,
        max_specialist_workers=args.max_specialist_workers,
    )
    daemon.run_forever()
    return None


def _cmd_event_list(args) -> list[dict[str, Any]]:
    return list_events(
        workspace_dir=args.workspace,
        project_id=args.project_id,
        actor=args.actor,
        action=args.action,
        subject_type=args.subject_type,
        limit=args.limit,
    )


def _load_json_arg(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {value}")
    return loaded


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_llm_client_from_env() -> OpenAICompatLLMClient:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("STARTEAM_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing DASHSCOPE_API_KEY (or legacy STARTEAM_API_KEY) for DataOS Agent CLI commands"
        )
    base_url = (
        os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("STARTEAM_BASE_URL")
        or DASHSCOPE_BASE_URL
    )
    model = (
        os.environ.get("DASHSCOPE_MODEL")
        or os.environ.get("STARTEAM_MODEL")
        or DEFAULT_MODEL
    )
    return OpenAICompatLLMClient(api_key=api_key, base_url=base_url, model=model)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
