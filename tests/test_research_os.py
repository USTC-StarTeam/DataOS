from __future__ import annotations

import json

import pytest

import dataos.cli as dataos_cli
from dataos.cli import main as cli_main
from dataos.research.agent import DataOSAgent
from dataos.research.orchestrator import run_project_step, run_ready_tasks
from dataos.research.scheduler import scheduler_tick
from dataos.research.store import ResearchStore
from dataos.research.worker import worker_run


class FakeResearchLLM:
    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0):
        del user_prompt, temperature
        if "hypothesis" in system_prompt:
            return {
                "proposal_type": "hypotheses",
                "payload": {
                    "hypotheses": [
                        {
                            "title": "Filter noisy samples",
                            "rationale": "Low-quality examples may hurt SFT.",
                        }
                    ]
                },
            }
        if "experiment_planner" in system_prompt:
            return {
                "proposal_type": "research_tasks",
                "payload": {
                    "tasks": [
                        {
                            "task_type": "experiment_run",
                            "priority": 5,
                            "spec": {"stub_result": {"score": 0.5, "name": "baseline"}},
                        }
                    ]
                },
            }
        if "operator_designer" in system_prompt:
            return {
                "proposal_type": "candidate_operator",
                "payload": {
                    "name": "quality_filter_candidate",
                    "spec": {"kind": "filter", "params_schema": {"threshold": "float"}},
                    "validation_task": {
                        "spec": {
                            "stub_result": {
                                "candidate_operator_status": "validated",
                                "validation_report": {"ok": True},
                            }
                        }
                    },
                },
            }
        if "critic" in system_prompt:
            return {
                "proposal_type": "critique",
                "payload": {"issues": [], "recommendation": "Proceed."},
            }
        if "evaluation_analyst" in system_prompt:
            return {
                "proposal_type": "analysis",
                "payload": {"summary": "Baseline completed."},
            }
        if "report" in system_prompt:
            return {
                "proposal_type": "report",
                "payload": {"summary": {"finding": "Stub campaign complete"}},
            }
        raise AssertionError(system_prompt)


class RoleEchoLLM:
    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0):
        del user_prompt, temperature
        if "hypothesis" in system_prompt:
            return {
                "proposal_type": "hypotheses",
                "payload": {"hypotheses": [{"title": "Echo hypothesis"}]},
            }
        if "critic" in system_prompt:
            return {
                "proposal_type": "critique",
                "payload": {"issues": []},
            }
        raise AssertionError(system_prompt)


class BadResearchLLM:
    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0):
        del system_prompt, user_prompt, temperature
        return {"proposal_type": "hypotheses", "payload": {"hypotheses": []}}


def _task_template():
    return {
        "task_id": "research-test",
        "name": "Research Test",
        "goal": "Improve data-centric SFT",
        "input_description": "HF dataset",
        "output_description": "artifacts",
        "constraints": {"dataset_format": "hf_datasets"},
        "budget": {"max_experiments": 2},
        "max_rounds": 1,
        "pipeline_hint": ["dedup"],
        "train_config": {"mode": "stub"},
        "eval_config": {"provider": "stub", "provider_config": {"score": 0.5}},
    }


def _write_task_template(tmp_path):
    path = tmp_path / "task.json"
    path.write_text(json.dumps(_task_template()), encoding="utf-8")
    return path


def test_research_store_candidate_operator_gate(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Improve data quality",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    candidate = store.create_candidate_operator(
        project.project_id,
        name="candidate_filter",
        spec={"kind": "filter"},
    )

    with pytest.raises(ValueError, match="validated or promoted"):
        store.submit_research_task(
            project.project_id,
            task_type="experiment_run",
            spec={
                "operator_refs": [
                    {"type": "candidate", "candidate_operator_id": candidate.candidate_operator_id}
                ],
                "stub_result": {"score": 0.1},
            },
        )

    store.update_candidate_operator_status(
        candidate.candidate_operator_id,
        "validated",
        validation_report={"ok": True},
    )
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={
            "operator_refs": [
                {"type": "candidate", "candidate_operator_id": candidate.candidate_operator_id}
            ],
            "stub_result": {"score": 0.2},
        },
    )
    assert task.status == "pending"


def test_research_cli_candidate_operator_status_flow(tmp_path, capsys):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Validate candidate operators",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    candidate = store.create_candidate_operator(
        project.project_id,
        name="candidate_filter",
        spec={"kind": "filter"},
    )

    cli_main(
        [
            "candidate",
            "list",
            "--workspace",
            str(tmp_path),
            "--project-id",
            project.project_id,
            "--status",
            "proposed",
        ]
    )
    listed = json.loads(capsys.readouterr().out)
    assert [item["candidate_operator_id"] for item in listed] == [
        candidate.candidate_operator_id
    ]

    cli_main(
        [
            "candidate",
            "set-status",
            "--workspace",
            str(tmp_path),
            candidate.candidate_operator_id,
            "--status",
            "validated",
            "--validation-report",
            '{"ok": true, "metric": 0.82}',
        ]
    )
    updated = json.loads(capsys.readouterr().out)
    assert updated["status"] == "validated"
    assert updated["validation_report"]["ok"] is True

    cli_main(
        [
            "candidate",
            "get",
            "--workspace",
            str(tmp_path),
            candidate.candidate_operator_id,
        ]
    )
    fetched = json.loads(capsys.readouterr().out)
    assert fetched["candidate_operator_id"] == candidate.candidate_operator_id

    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={
            "operator_refs": [
                {"type": "candidate", "candidate_operator_id": candidate.candidate_operator_id}
            ],
            "stub_result": {"score": 0.2},
        },
    )
    assert task.status == "pending"

    cli_main(
        [
            "event",
            "list",
            "--workspace",
            str(tmp_path),
            "--project-id",
            project.project_id,
            "--action",
            "candidate_operator_validated",
        ]
    )
    events = json.loads(capsys.readouterr().out)
    assert events[0]["subject_id"] == candidate.candidate_operator_id


def test_research_store_enforces_experiment_run_budget(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Respect project budget",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    first = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    store.submit_research_task(
        project.project_id,
        task_type="eval_run",
        spec={"stub_result": {"score": 0.2}},
    )

    with pytest.raises(ValueError, match="budget exhausted"):
        store.submit_research_task(
            project.project_id,
            task_type="experiment_run",
            spec={"stub_result": {"score": 0.3}},
        )

    summary = store.export_project_summary(project.project_id)
    assert summary["experiment_run_budget"] == {"limit": 1, "used": 1, "remaining": 0}

    store.cancel_research_task(first.task_id, reason="Free budget.")
    replacement = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.4}},
    )
    assert replacement.status == "pending"

    with pytest.raises(ValueError, match="budget exhausted"):
        store.requeue_research_task(first.task_id)


def test_research_project_status_controls_task_submission_and_scheduler(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Pause project work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    store.add_resource(
        resource_id="local-single",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=1,
    )
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )

    paused = store.update_project_status(project.project_id, "paused")
    assert paused.status == "paused"
    with pytest.raises(ValueError, match="Project is not active"):
        store.submit_research_task(
            project.project_id,
            task_type="experiment_run",
            spec={"stub_result": {"score": 0.2}},
        )
    report = scheduler_tick(tmp_path)
    assert report.scheduled_task_ids == []
    assert store.get_research_task(task.task_id).status == "pending"

    store.update_project_status(project.project_id, "running")
    report = scheduler_tick(tmp_path)
    assert report.scheduled_task_ids == [task.task_id]


def test_research_project_auto_moves_to_running_when_work_starts(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Track active project state",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    assert store.get_project(project.project_id).status == "created"

    store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )

    assert store.get_project(project.project_id).status == "running"
    running_events = store.list_events(
        project_id=project.project_id,
        action="project_running",
    )
    assert len(running_events) == 1
    assert running_events[0].actor == "human"

    store.submit_research_task(
        project.project_id,
        task_type="eval_run",
        spec={"stub_result": {"score": 0.2}},
    )
    running_events = store.list_events(
        project_id=project.project_id,
        action="project_running",
    )
    assert len(running_events) == 1


def test_dataos_agent_manager_marks_project_running_when_it_spawns_work(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Track active agent work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    agent = DataOSAgent(tmp_path)

    decision = agent.manager_tick(project.project_id)

    assert decision.action == "spawned_initial_agent_tasks"
    assert store.get_project(project.project_id).status == "running"
    running_events = store.list_events(
        project_id=project.project_id,
        action="project_running",
    )
    assert len(running_events) == 1
    assert running_events[0].actor == "research_manager"


def test_research_store_blocks_new_research_content_for_inactive_project(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Freeze inactive project writes",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    agent_task = store.create_agent_task(
        project.project_id,
        role="hypothesis",
        input_context={"project_id": project.project_id},
    )
    proposal = store.create_agent_proposal(
        project_id=project.project_id,
        agent_task_id=agent_task.agent_task_id,
        role="hypothesis",
        proposal_type="hypotheses",
        payload={"hypotheses": [{"title": "Already proposed"}]},
    )
    store.update_project_status(project.project_id, "paused")

    with pytest.raises(ValueError, match="Project is not active"):
        store.add_hypothesis(project.project_id, title="Too late", rationale="")
    with pytest.raises(ValueError, match="Project is not active"):
        store.create_candidate_operator(
            project.project_id,
            name="late_candidate",
            spec={"kind": "filter"},
        )
    with pytest.raises(ValueError, match="Project is not active"):
        store.create_agent_task(project.project_id, role="critic", input_context={})
    with pytest.raises(ValueError, match="Project is not active"):
        store.create_agent_proposal(
            project_id=project.project_id,
            agent_task_id=agent_task.agent_task_id,
            role="critic",
            proposal_type="critique",
            payload={"issues": []},
        )

    agent = DataOSAgent(tmp_path)
    with pytest.raises(ValueError, match="Project is not active"):
        agent.accept_proposal(proposal.proposal_id)
    assert store.get_agent_proposal(proposal.proposal_id).status == "proposed"

    agent.reject_proposal(proposal.proposal_id, "Paused for review.")
    rejected = store.get_agent_proposal(proposal.proposal_id)
    assert rejected.status == "rejected"
    assert rejected.reason == "Paused for review."


def test_research_store_validates_agent_proposal_source(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Validate proposal provenance",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    other_project = store.create_project(
        goal="Other proposal project",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    agent_task = store.create_agent_task(
        project.project_id,
        role="hypothesis",
        input_context={"project_id": project.project_id},
    )

    with pytest.raises(ValueError, match="same project"):
        store.create_agent_proposal(
            project_id=other_project.project_id,
            agent_task_id=agent_task.agent_task_id,
            role="hypothesis",
            proposal_type="hypotheses",
            payload={"hypotheses": [{"title": "Wrong project"}]},
        )
    with pytest.raises(ValueError, match="role must match"):
        store.create_agent_proposal(
            project_id=project.project_id,
            agent_task_id=agent_task.agent_task_id,
            role="critic",
            proposal_type="critique",
            payload={"issues": []},
        )


def test_research_agent_task_and_proposal_terminal_statuses_are_guarded(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Protect agent state machines",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    completed_agent_task = store.create_agent_task(
        project.project_id,
        role="hypothesis",
        input_context={},
    )
    store.update_agent_task_status(completed_agent_task.agent_task_id, "completed")
    with pytest.raises(ValueError, match="Cannot transition terminal agent task status"):
        store.update_agent_task_status(completed_agent_task.agent_task_id, "running")

    retryable_agent_task = store.create_agent_task(
        project.project_id,
        role="critic",
        input_context={},
    )
    store.update_agent_task_status(retryable_agent_task.agent_task_id, "failed")
    retried = store.update_agent_task_status(retryable_agent_task.agent_task_id, "running")
    assert retried.status == "running"

    proposal_task = store.create_agent_task(
        project.project_id,
        role="experiment_planner",
        input_context={},
    )
    proposal = store.create_agent_proposal(
        project_id=project.project_id,
        agent_task_id=proposal_task.agent_task_id,
        role="experiment_planner",
        proposal_type="research_tasks",
        payload={
            "tasks": [
                {"task_type": "experiment_run", "spec": {"stub_result": {"score": 0.1}}}
            ]
        },
    )
    store.update_agent_proposal_status(proposal.proposal_id, "accepted")
    with pytest.raises(ValueError, match="Cannot transition terminal agent proposal status"):
        store.update_agent_proposal_status(proposal.proposal_id, "rejected")


def test_research_project_cancel_cancels_open_work_and_releases_leases(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Cancel project work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 3},
    )
    store.add_resource(
        resource_id="local-pair",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=2,
    )
    scheduled_task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
        priority=3,
    )
    running_task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
        priority=2,
    )
    pending_task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.3}},
        priority=1,
    )
    scheduler_tick(tmp_path, max_tasks=2)
    scheduled = store.get_research_task(scheduled_task.task_id)
    running = store.get_research_task(running_task.task_id)
    assert scheduled.lease_id is not None
    assert running.lease_id is not None
    store.update_task_status(
        running.task_id,
        "running",
        actor="worker",
        lease_id=running.lease_id,
    )
    pending_agent_task = store.create_agent_task(
        project.project_id,
        role="hypothesis",
        input_context={},
    )
    running_agent_task = store.create_agent_task(
        project.project_id,
        role="critic",
        input_context={},
    )
    store.update_agent_task_status(running_agent_task.agent_task_id, "running")

    cancelled = store.update_project_status(project.project_id, "cancelled")

    assert cancelled.status == "cancelled"
    for task_id in [scheduled_task.task_id, running_task.task_id, pending_task.task_id]:
        task = store.get_research_task(task_id)
        assert task.status == "cancelled"
        assert task.error == "Project cancelled."
    assert store.get_lease(scheduled.lease_id).status == "released"
    assert store.get_lease(running.lease_id).status == "released"
    assert store.active_lease_count("local-pair") == 0
    assert store.get_agent_task(pending_agent_task.agent_task_id).status == "cancelled"
    assert store.get_agent_task(running_agent_task.agent_task_id).status == "cancelled"


def test_research_project_complete_rejects_open_work(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Complete only after work closes",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    agent_task = store.create_agent_task(project.project_id, role="critic", input_context={})

    with pytest.raises(ValueError, match="Cannot complete project with open work"):
        store.update_project_status(project.project_id, "completed")

    store.cancel_research_task(task.task_id)
    store.update_agent_task_status(agent_task.agent_task_id, "cancelled")
    completed = store.update_project_status(project.project_id, "completed")
    assert completed.status == "completed"


def test_research_project_summary_completion_reuses_lifecycle_guard(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Do not complete through summary while work is open",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )

    with pytest.raises(ValueError, match="Cannot complete project with open work"):
        store.update_project_summary(
            project.project_id,
            {"finding": "Too early."},
            status="completed",
            actor="report",
        )
    assert store.get_project(project.project_id).status == "running"

    store.cancel_research_task(task.task_id)
    completed = store.update_project_summary(
        project.project_id,
        {"finding": "Closed."},
        status="completed",
        actor="report",
    )
    assert completed.status == "completed"
    assert completed.summary == {"finding": "Closed."}
    events = store.list_events(project_id=project.project_id, action="project_completed")
    assert events[0].actor == "report"


def test_research_project_terminal_status_cannot_be_reopened(tmp_path):
    store = ResearchStore(tmp_path)
    cancelled_project = store.create_project(
        goal="Cancelled projects stay closed",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    store.update_project_status(cancelled_project.project_id, "cancelled")
    with pytest.raises(ValueError, match="Cannot transition terminal project status"):
        store.update_project_status(cancelled_project.project_id, "running")
    assert store.get_project(cancelled_project.project_id).status == "cancelled"

    completed_project = store.create_project(
        goal="Completed projects stay closed",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    store.update_project_status(completed_project.project_id, "completed")
    with pytest.raises(ValueError, match="Cannot transition terminal project status"):
        store.update_project_summary(
            completed_project.project_id,
            {"finding": "Reopen through summary."},
            status="running",
        )
    assert store.get_project(completed_project.project_id).status == "completed"


def test_dataos_agent_report_proposal_cannot_complete_open_project(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Reject premature reports",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    agent_task = store.create_agent_task(project.project_id, role="report", input_context={})
    proposal = store.create_agent_proposal(
        project_id=project.project_id,
        agent_task_id=agent_task.agent_task_id,
        role="report",
        proposal_type="report",
        payload={"summary": {"finding": "Too early."}, "project_status": "completed"},
    )
    agent = DataOSAgent(tmp_path)

    with pytest.raises(ValueError, match="Cannot complete project with open work"):
        agent.accept_proposal(proposal.proposal_id)
    assert store.get_project(project.project_id).status == "running"
    assert store.get_agent_proposal(proposal.proposal_id).status == "proposed"


def test_dataos_agent_rejects_research_task_proposal_that_exceeds_budget(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Reject over-budget proposals",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    agent_task = store.create_agent_task(
        project.project_id,
        role="experiment_planner",
        input_context={"project_id": project.project_id},
    )
    proposal = store.create_agent_proposal(
        project_id=project.project_id,
        agent_task_id=agent_task.agent_task_id,
        role="experiment_planner",
        proposal_type="research_tasks",
        payload={
            "tasks": [
                {"task_type": "experiment_run", "spec": {"stub_result": {"score": 0.1}}},
                {"task_type": "experiment_run", "spec": {"stub_result": {"score": 0.2}}},
            ]
        },
    )
    agent = DataOSAgent(tmp_path)

    with pytest.raises(ValueError, match="budget exhausted"):
        agent.accept_proposal(proposal.proposal_id)
    assert not store.list_research_tasks(project.project_id)
    assert store.get_agent_proposal(proposal.proposal_id).status == "proposed"


def test_dataos_agent_prevalidates_research_task_proposal_before_commit(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Reject invalid task proposals atomically",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    agent_task = store.create_agent_task(
        project.project_id,
        role="experiment_planner",
        input_context={"project_id": project.project_id},
    )
    proposal = store.create_agent_proposal(
        project_id=project.project_id,
        agent_task_id=agent_task.agent_task_id,
        role="experiment_planner",
        proposal_type="research_tasks",
        payload={
            "tasks": [
                {"task_type": "experiment_run", "spec": {"stub_result": {"score": 0.1}}},
                {
                    "task_type": "eval_run",
                    "spec": {"stub_result": {"score": 0.2}},
                    "dependencies": ["missing-task"],
                },
            ]
        },
    )
    agent = DataOSAgent(tmp_path)

    with pytest.raises(KeyError, match="Unknown task_id"):
        agent.accept_proposal(proposal.proposal_id)
    assert not store.list_research_tasks(project.project_id)


def test_research_scheduler_dependencies_and_worker_stub(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Run parallel research tasks",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    store.add_resource(
        resource_id="gpu-local",
        host="local",
        kind="local",
        gpu_ids=["0"],
        max_concurrent_tasks=1,
    )
    first = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.3}},
        resource_request={"gpus": 1},
        priority=1,
    )
    second = store.submit_research_task(
        project.project_id,
        task_type="eval_run",
        spec={"stub_result": {"score": 0.4}},
        dependencies=[first.task_id],
        priority=2,
    )

    report = scheduler_tick(tmp_path)
    assert report.scheduled_task_ids == [first.task_id]
    scheduled_first = store.get_research_task(first.task_id)
    assert scheduled_first.status == "scheduled"
    assert scheduled_first.started_at is None
    assert store.get_research_task(second.task_id).status == "pending"

    worker_report = worker_run(first.task_id, tmp_path)
    assert worker_report.status == "completed"
    assert store.get_research_task(first.task_id).started_at is not None

    report = scheduler_tick(tmp_path)
    assert report.scheduled_task_ids == [second.task_id]
    assert worker_run(second.task_id, tmp_path).result["score"] == 0.4
    assert store.export_project_summary(project.project_id)["best_score"] == 0.4


def test_research_pause_releases_scheduled_lease_before_worker_starts(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Pause before worker starts",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.3}},
    )
    scheduler_tick(tmp_path)
    scheduled = store.get_research_task(task.task_id)
    assert scheduled.status == "scheduled"
    assert scheduled.lease_id is not None

    store.update_project_status(project.project_id, "paused")
    with pytest.raises(ValueError, match="running or scheduled"):
        worker_run(task.task_id, tmp_path)

    requeued = store.get_research_task(task.task_id)
    assert requeued.status == "pending"
    assert requeued.started_at is None
    assert requeued.lease_id is None
    assert requeued.error == "Project paused."
    assert store.get_lease(scheduled.lease_id).status == "released"
    assert store.active_lease_count("local-default") == 0


def test_research_store_validates_task_dependencies(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Validate dependencies",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    other_project = store.create_project(
        goal="Other dependency project",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    upstream = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    other_task = store.submit_research_task(
        other_project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
    )
    dependent = store.submit_research_task(
        project.project_id,
        task_type="eval_run",
        spec={"stub_result": {"score": 0.3}},
        dependencies=[upstream.task_id],
    )
    assert dependent.dependencies == [upstream.task_id]

    with pytest.raises(KeyError, match="Unknown task_id"):
        store.submit_research_task(
            project.project_id,
            task_type="eval_run",
            spec={"stub_result": {"score": 0.4}},
            dependencies=["missing-task"],
        )
    with pytest.raises(ValueError, match="same project"):
        store.submit_research_task(
            project.project_id,
            task_type="eval_run",
            spec={"stub_result": {"score": 0.5}},
            dependencies=[other_task.task_id],
        )


def test_research_scheduler_blocks_tasks_with_failed_dependencies(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Propagate failed dependency state",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    upstream = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    dependent = store.submit_research_task(
        project.project_id,
        task_type="eval_run",
        spec={"stub_result": {"score": 0.2}},
        dependencies=[upstream.task_id],
    )
    downstream = store.submit_research_task(
        project.project_id,
        task_type="analysis_task",
        spec={"stub_result": {"summary": "never runs"}},
        dependencies=[dependent.task_id],
    )
    store.update_task_status(upstream.task_id, "failed", actor="worker", error="Boom.")

    report = scheduler_tick(tmp_path)

    assert report.scheduled_task_ids == []
    assert report.blocked_task_ids == [dependent.task_id, downstream.task_id]
    blocked_dependent = store.get_research_task(dependent.task_id)
    blocked_downstream = store.get_research_task(downstream.task_id)
    assert blocked_dependent.status == "blocked"
    assert blocked_dependent.error == f"Blocked by dependency status: {upstream.task_id}=failed"
    assert blocked_downstream.status == "blocked"
    assert blocked_downstream.error == (
        f"Blocked by dependency status: {dependent.task_id}=blocked"
    )
    summary = store.export_project_summary(project.project_id)
    assert summary["task_status_counts"]["failed"] == 1
    assert summary["task_status_counts"]["blocked"] == 2
    assert summary["task_status_counts"]["pending"] == 0


def test_research_scheduler_unblocks_tasks_when_dependencies_recover(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Recover blocked dependency branches",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    upstream = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    dependent = store.submit_research_task(
        project.project_id,
        task_type="eval_run",
        spec={"stub_result": {"score": 0.2}},
        dependencies=[upstream.task_id],
    )
    downstream = store.submit_research_task(
        project.project_id,
        task_type="analysis_task",
        spec={"stub_result": {"summary": "eventually runs"}},
        dependencies=[dependent.task_id],
    )
    store.update_task_status(upstream.task_id, "failed", actor="worker", error="Boom.")
    report = scheduler_tick(tmp_path)
    assert report.blocked_task_ids == [dependent.task_id, downstream.task_id]

    store.requeue_research_task(upstream.task_id, actor="human", reason="Retry upstream.")
    report = scheduler_tick(tmp_path)

    assert report.unblocked_task_ids == [dependent.task_id, downstream.task_id]
    assert report.blocked_task_ids == []
    assert report.scheduled_task_ids == [upstream.task_id]
    assert store.get_research_task(dependent.task_id).status == "pending"
    assert store.get_research_task(dependent.task_id).error is None
    assert store.get_research_task(downstream.task_id).status == "pending"
    assert store.get_research_task(downstream.task_id).error is None

    worker_run(upstream.task_id, tmp_path)
    report = scheduler_tick(tmp_path)
    assert report.scheduled_task_ids == [dependent.task_id]


def test_research_scheduler_does_not_block_dependencies_for_inactive_project(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Paused projects are not mutated by scheduler",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    upstream = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    dependent = store.submit_research_task(
        project.project_id,
        task_type="eval_run",
        spec={"stub_result": {"score": 0.2}},
        dependencies=[upstream.task_id],
    )
    store.update_task_status(upstream.task_id, "failed", actor="worker", error="Boom.")
    store.update_project_status(project.project_id, "paused")

    report = scheduler_tick(tmp_path, project_id=project.project_id)

    assert report.blocked_task_ids == []
    assert store.get_research_task(dependent.task_id).status == "pending"


def test_research_scheduler_does_not_unblock_dependencies_for_inactive_project(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Paused projects do not recover blocked tasks",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    upstream = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    dependent = store.submit_research_task(
        project.project_id,
        task_type="eval_run",
        spec={"stub_result": {"score": 0.2}},
        dependencies=[upstream.task_id],
    )
    store.update_task_status(upstream.task_id, "failed", actor="worker", error="Boom.")
    scheduler_tick(tmp_path)
    assert store.get_research_task(dependent.task_id).status == "blocked"
    store.requeue_research_task(upstream.task_id, actor="human", reason="Retry upstream.")
    store.update_project_status(project.project_id, "paused")

    report = scheduler_tick(tmp_path, project_id=project.project_id)

    assert report.unblocked_task_ids == []
    assert store.get_research_task(dependent.task_id).status == "blocked"


def test_research_scheduler_max_tasks_counts_successful_schedules(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Skip unschedulable tasks",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    store.add_resource(
        resource_id="cpu-local",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=1,
    )
    blocked = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
        resource_request={"gpus": 1},
        priority=10,
    )
    runnable = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
        priority=1,
    )

    report = scheduler_tick(tmp_path, max_tasks=1)
    assert report.skipped_task_ids == [blocked.task_id]
    assert report.scheduled_task_ids == [runnable.task_id]
    assert store.get_research_task(blocked.task_id).status == "pending"
    assert store.get_research_task(runnable.task_id).status == "scheduled"


def test_research_scheduler_respects_project_concurrency_limit(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Limit project concurrency",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 3},
        constraints={"max_concurrent_tasks": 1},
    )
    store.add_resource(
        resource_id="local-pair",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=2,
    )
    first = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
        priority=3,
    )
    second = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
        priority=2,
    )

    report = scheduler_tick(tmp_path, max_tasks=2)
    assert report.scheduled_task_ids == [first.task_id]
    assert report.skipped_task_ids == [second.task_id]
    assert store.get_project_concurrency(project.project_id) == {
        "limit": 1,
        "active": 1,
        "remaining": 0,
    }
    assert store.export_project_summary(project.project_id)["project_concurrency"] == {
        "limit": 1,
        "active": 1,
        "remaining": 0,
    }

    worker_run(first.task_id, tmp_path)
    report = scheduler_tick(tmp_path, max_tasks=2)
    assert report.scheduled_task_ids == [second.task_id]


def test_research_scheduler_project_concurrency_does_not_block_other_projects(tmp_path):
    store = ResearchStore(tmp_path)
    first_project = store.create_project(
        goal="Limited project",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
        constraints={"max_concurrent_tasks": 1},
    )
    second_project = store.create_project(
        goal="Other project",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
        constraints={"max_concurrent_tasks": 1},
    )
    store.add_resource(
        resource_id="local-pair",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=2,
    )
    first_a = store.submit_research_task(
        first_project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
        priority=3,
    )
    first_b = store.submit_research_task(
        first_project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
        priority=2,
    )
    second = store.submit_research_task(
        second_project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.3}},
        priority=1,
    )

    report = scheduler_tick(tmp_path, max_tasks=2)
    assert report.scheduled_task_ids == [first_a.task_id, second.task_id]
    assert report.skipped_task_ids == [first_b.task_id]


def test_research_store_scheduling_claim_is_atomic(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Claim tasks once",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    resource = store.add_resource(
        resource_id="local-single",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=1,
    )
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )

    scheduled = store.schedule_research_task(task, resource)
    assert scheduled.status == "scheduled"

    with pytest.raises(ValueError, match="no longer schedulable"):
        store.schedule_research_task(task, resource)
    assert store.active_lease_count(resource.resource_id) == 1
    assert len(store.list_leases(resource_id=resource.resource_id)) == 1


def test_research_task_cancel_releases_scheduled_lease(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Cancel scheduled work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    store.add_resource(
        resource_id="local-single",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=1,
    )
    first = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
        priority=2,
    )
    second = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
        priority=1,
    )

    report = scheduler_tick(tmp_path, max_tasks=1)
    assert report.scheduled_task_ids == [first.task_id]
    lease_id = store.get_research_task(first.task_id).lease_id
    assert lease_id is not None
    assert store.active_lease_count("local-single") == 1

    cancelled = store.cancel_research_task(
        first.task_id,
        actor="human",
        reason="Superseded by a better run.",
    )
    assert cancelled.status == "cancelled"
    assert cancelled.error == "Superseded by a better run."
    assert store.get_lease(lease_id).status == "released"
    assert store.active_lease_count("local-single") == 0

    report = scheduler_tick(tmp_path, max_tasks=1)
    assert report.scheduled_task_ids == [second.task_id]


def test_research_task_requeue_releases_lease_and_clears_execution_state(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Requeue stuck work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    store.add_resource(
        resource_id="local-single",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=1,
    )
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    scheduler_tick(tmp_path)
    scheduled = store.get_research_task(task.task_id)
    assert scheduled.status == "scheduled"
    assert scheduled.lease_id is not None

    requeued = store.requeue_research_task(
        task.task_id,
        actor="human",
        reason="Worker did not start.",
    )

    assert requeued.status == "pending"
    assert requeued.lease_id is None
    assert requeued.result == {}
    assert requeued.started_at is None
    assert requeued.finished_at is None
    assert requeued.error == "Worker did not start."
    assert store.active_lease_count("local-single") == 0
    assert store.get_lease(scheduled.lease_id).status == "released"


def test_research_task_terminal_status_requires_requeue_for_retry(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Protect terminal task status",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    failed_task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    store.update_task_status(failed_task.task_id, "failed", actor="worker", error="Boom.")

    with pytest.raises(ValueError, match="Cannot transition terminal research task status"):
        store.update_task_status(failed_task.task_id, "pending", actor="human")

    requeued = store.requeue_research_task(
        failed_task.task_id,
        actor="human",
        reason="Retry through the guarded path.",
    )
    assert requeued.status == "pending"

    completed_task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
    )
    scheduler_tick(tmp_path)
    worker_run(completed_task.task_id, tmp_path)
    with pytest.raises(ValueError, match="Cannot transition terminal research task status"):
        store.update_task_status(completed_task.task_id, "running", actor="human")


def test_research_task_requeue_requires_active_project(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Do not requeue inactive project work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    store.update_task_status(task.task_id, "failed", actor="worker", error="Transient.")
    store.update_project_status(project.project_id, "paused")

    with pytest.raises(ValueError, match="Project is not active"):
        store.requeue_research_task(task.task_id)
    assert store.get_research_task(task.task_id).status == "failed"

    store.update_project_status(project.project_id, "running")
    store.update_project_status(project.project_id, "cancelled")
    with pytest.raises(ValueError, match="Project is not active"):
        store.requeue_research_task(task.task_id)
    assert store.get_research_task(task.task_id).status == "failed"


def test_research_store_direct_schedule_requires_active_project(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Do not schedule inactive project work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    resource = store.add_resource(
        resource_id="local-single",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=1,
    )
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    store.update_project_status(project.project_id, "paused")

    with pytest.raises(ValueError, match="Project is not active"):
        store.schedule_research_task(task, resource)
    assert not store.list_leases(status="active", project_id=project.project_id)
    assert store.get_research_task(task.task_id).status == "pending"


def test_research_resource_leases_can_be_filtered_by_project_and_status(tmp_path, capsys):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Inspect resource leases",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    other_project = store.create_project(
        goal="Inspect another project",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    store.add_resource(
        resource_id="local-pair",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=2,
    )
    first = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    second = store.submit_research_task(
        other_project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
    )
    scheduler_tick(tmp_path, max_tasks=2)

    leases = store.list_leases(project_id=project.project_id, status="active")
    assert [lease.task_id for lease in leases] == [first.task_id]

    cli_main(
        [
            "resource",
            "leases",
            "--workspace",
            str(tmp_path),
            "--project-id",
            other_project.project_id,
            "--status",
            "active",
        ]
    )
    cli_leases = json.loads(capsys.readouterr().out)
    assert [lease["task_id"] for lease in cli_leases] == [second.task_id]

    worker_run(first.task_id, tmp_path)
    released = store.list_leases(resource_id="local-pair", status="released")
    assert [lease.task_id for lease in released] == [first.task_id]


def test_research_scheduler_recovers_stale_tasks_with_project_filter(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Recover one project",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    other_project = store.create_project(
        goal="Leave this project alone",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    store.add_resource(
        resource_id="local-pair",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=2,
    )
    first = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    second = store.submit_research_task(
        other_project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
    )
    scheduler_tick(tmp_path, max_tasks=2)
    assert store.get_research_task(first.task_id).status == "scheduled"
    assert store.get_research_task(second.task_id).status == "scheduled"

    from dataos.research.scheduler import recover_stale_tasks

    report = recover_stale_tasks(
        tmp_path,
        older_than_seconds=0,
        project_id=project.project_id,
    )
    assert report.requeued_task_ids == [first.task_id]
    assert store.get_research_task(first.task_id).status == "pending"
    assert store.get_research_task(second.task_id).status == "scheduled"


def test_research_orchestrator_runs_ready_tasks_concurrently(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Run ready tasks",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 3},
    )
    store.add_resource(
        resource_id="local-batch",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=3,
    )
    first = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.1}},
    )
    second = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.2}},
    )
    dependent = store.submit_research_task(
        project.project_id,
        task_type="eval_run",
        spec={"stub_result": {"score": 0.3}},
        dependencies=[first.task_id, second.task_id],
    )

    report = run_ready_tasks(tmp_path, max_workers=2)
    assert {item.task_id for item in report.worker_reports} == {first.task_id, second.task_id}
    assert store.get_research_task(dependent.task_id).status == "pending"

    report = run_ready_tasks(tmp_path, max_workers=2)
    assert [item.task_id for item in report.worker_reports] == [dependent.task_id]
    assert store.get_research_task(dependent.task_id).status == "completed"


def test_research_orchestrator_project_step_runs_agent_and_ready_tasks(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Advance campaign and queue together",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 4},
    )
    existing_task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.9}},
    )
    other_project = store.create_project(
        goal="Do not run this project",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    other_task = store.submit_research_task(
        other_project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 1.0}},
        priority=10,
    )

    report = run_project_step(
        tmp_path,
        project.project_id,
        llm_client=FakeResearchLLM(),
        agent_max_workers=4,
        task_max_workers=2,
    )

    assert report.agent_cycle is not None
    assert report.agent_cycle.final_decision.action == "accepted_proposals"
    assert report.task_run is not None
    assert {item.task_id for item in report.task_run.worker_reports} == {
        existing_task.task_id
    }
    assert report.post_agent_task_run is not None
    assert report.post_agent_task_run.worker_reports
    completed_tasks = store.list_research_tasks(project.project_id, status="completed")
    assert len(completed_tasks) >= 2
    assert store.get_research_task(other_task.task_id).status == "pending"


def test_dataos_agent_manager_tick_does_not_require_llm_client(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Prepare research campaign",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 4},
    )
    agent = DataOSAgent(tmp_path)

    decision = agent.manager_tick(project.project_id)
    assert decision.action == "spawned_initial_agent_tasks"
    assert len(decision.agent_task_ids) == 4

    with pytest.raises(ValueError, match="llm_client is required"):
        agent.run_agent_task(decision.agent_task_ids[0])
    assert store.get_agent_task(decision.agent_task_ids[0]).status == "failed"
    assert not store.list_agent_proposals(project.project_id)


def test_dataos_agent_manager_noops_for_inactive_project(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Pause agent work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    store.update_project_status(project.project_id, "paused")
    agent = DataOSAgent(tmp_path, FakeResearchLLM())

    decision = agent.manager_tick(project.project_id)
    assert decision.action == "project_inactive"
    assert decision.notes == ["Project status is paused."]
    assert not store.list_agent_tasks(project.project_id)


def test_dataos_agent_does_not_run_pending_tasks_for_inactive_project(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Pause existing agent work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    agent = DataOSAgent(tmp_path, FakeResearchLLM())
    decision = agent.manager_tick(project.project_id)
    assert decision.action == "spawned_initial_agent_tasks"
    store.update_project_status(project.project_id, "paused")

    with pytest.raises(ValueError, match="Project is not active"):
        agent.run_agent_task(decision.agent_task_ids[0])
    with pytest.raises(ValueError, match="Project is not active"):
        agent.run_pending_agent_tasks(project.project_id)

    cycle = agent.run_cycle(project.project_id)
    assert cycle.first_decision.action == "project_inactive"
    assert not cycle.agent_run.completed_agent_task_ids
    assert cycle.final_decision.action == "project_inactive"


def test_dataos_agent_parallel_proposal_flow(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Improve GSM8K SFT performance",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 4},
    )
    agent = DataOSAgent(tmp_path, FakeResearchLLM())

    decision = agent.manager_tick(project.project_id)
    assert decision.action == "spawned_initial_agent_tasks"
    assert len(decision.agent_task_ids) == 4

    for agent_task_id in decision.agent_task_ids:
        agent.run_agent_task(agent_task_id)
    decision = agent.manager_tick(project.project_id)
    assert decision.action == "accepted_proposals"
    assert decision.research_task_ids
    assert len(store.list_hypotheses(project.project_id)) == 1

    scheduler_tick(tmp_path, max_tasks=10)
    scheduled = store.list_research_tasks(project.project_id, status="scheduled")
    assert scheduled
    for task in scheduled:
        worker_run(task.task_id, tmp_path)

    decision = agent.manager_tick(project.project_id)
    assert decision.action == "spawned_result_agent_tasks"
    roles = {store.get_agent_task(agent_task_id).role for agent_task_id in decision.agent_task_ids}
    assert {"evaluation_analyst", "critic", "operator_designer"}.issubset(roles)


def test_dataos_agent_cycle_spawns_runs_and_merges_parallel_specialists(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Run one agent cycle",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 4},
    )
    agent = DataOSAgent(tmp_path, FakeResearchLLM())

    report = agent.run_cycle(project.project_id, max_workers=4)
    assert report.first_decision.action == "spawned_initial_agent_tasks"
    assert len(report.agent_run.completed_agent_task_ids) == 4
    assert report.final_decision.action == "accepted_proposals"
    assert report.final_decision.research_task_ids
    assert {task.status for task in store.list_agent_tasks(project.project_id)} == {"completed"}
    assert store.list_research_tasks(project.project_id)
    assert store.list_candidate_operators(project.project_id)


def test_dataos_agent_result_analysis_is_tracked_per_completed_task(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Analyze asynchronous results",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 4},
    )
    store.add_hypothesis(project.project_id, title="Baseline matters", rationale="Track results.")
    agent = DataOSAgent(tmp_path, FakeResearchLLM())

    first = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.3}},
    )
    scheduler_tick(tmp_path)
    worker_run(first.task_id, tmp_path)

    decision = agent.manager_tick(project.project_id)
    assert decision.action == "spawned_result_agent_tasks"
    agent.run_pending_agent_tasks(
        project.project_id,
        max_workers=1,
    )
    decision = agent.manager_tick(project.project_id)
    assert decision.action == "accepted_proposals"
    analysis_events = store.list_events(
        project_id=project.project_id,
        action="analysis_recorded",
    )
    assert analysis_events[0].payload["covered_task_ids"] == [first.task_id]

    second = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.4}},
    )
    scheduler_tick(tmp_path)
    worker_run(second.task_id, tmp_path)

    decision = agent.manager_tick(project.project_id)
    assert decision.action == "spawned_result_agent_tasks"


def test_dataos_agent_manager_does_not_duplicate_open_result_analysis_tasks(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Avoid duplicate result analysis",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    store.add_hypothesis(project.project_id, title="Baseline matters", rationale="Track results.")
    agent = DataOSAgent(tmp_path, FakeResearchLLM())
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.4}},
    )
    scheduler_tick(tmp_path)
    worker_run(task.task_id, tmp_path)

    first = agent.manager_tick(project.project_id)
    assert first.action == "spawned_result_agent_tasks"

    second = agent.manager_tick(project.project_id)
    assert second.action == "waiting_for_agent_tasks"
    assert second.agent_task_ids == first.agent_task_ids
    result_analysis_tasks = [
        task
        for task in store.list_agent_tasks(project.project_id)
        if task.input_context.get("context_scope") == "result_analysis"
    ]
    assert len(result_analysis_tasks) == 3


def test_dataos_agent_manager_does_not_duplicate_open_report_task(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Avoid duplicate report tasks",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    store.add_hypothesis(project.project_id, title="Baseline matters", rationale="Track results.")
    agent = DataOSAgent(tmp_path, FakeResearchLLM())
    first_task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.3}},
    )
    second_task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.5}},
    )
    scheduler_tick(tmp_path, max_tasks=2)
    worker_run(first_task.task_id, tmp_path)
    worker_run(second_task.task_id, tmp_path)
    store.append_event(
        project_id=project.project_id,
        actor="evaluation_analyst",
        action="analysis_recorded",
        subject_type="agent_proposal",
        subject_id="manual-analysis",
        payload={"covered_task_ids": [first_task.task_id, second_task.task_id]},
    )

    first = agent.manager_tick(project.project_id)
    assert first.action == "spawned_report_agent_task"

    second = agent.manager_tick(project.project_id)
    assert second.action == "waiting_for_agent_tasks"
    assert second.agent_task_ids == first.agent_task_ids
    report_tasks = [
        task
        for task in store.list_agent_tasks(project.project_id)
        if task.input_context.get("context_scope") == "final_report"
    ]
    assert len(report_tasks) == 1


def test_dataos_agent_report_ready_when_single_run_budget_is_exhausted(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Report after one budgeted run",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    store.add_hypothesis(project.project_id, title="One run is enough", rationale="V1 budget.")
    agent = DataOSAgent(tmp_path, FakeResearchLLM())
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.7}},
    )
    scheduler_tick(tmp_path)
    worker_run(task.task_id, tmp_path)
    store.append_event(
        project_id=project.project_id,
        actor="evaluation_analyst",
        action="analysis_recorded",
        subject_type="agent_proposal",
        subject_id="manual-analysis",
        payload={"covered_task_ids": [task.task_id]},
    )

    decision = agent.manager_tick(project.project_id)
    assert decision.action == "spawned_report_agent_task"

    second = agent.manager_tick(project.project_id)
    assert second.action == "waiting_for_agent_tasks"
    assert second.agent_task_ids == decision.agent_task_ids


def test_dataos_agent_report_waits_for_open_research_tasks(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Do not report while work is open",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    store.add_hypothesis(project.project_id, title="One run is enough", rationale="V1 budget.")
    agent = DataOSAgent(tmp_path, FakeResearchLLM())
    task = store.submit_research_task(
        project.project_id,
        task_type="experiment_run",
        spec={"stub_result": {"score": 0.7}},
    )
    scheduler_tick(tmp_path)
    store.append_event(
        project_id=project.project_id,
        actor="evaluation_analyst",
        action="analysis_recorded",
        subject_type="agent_proposal",
        subject_id="manual-analysis",
        payload={"covered_task_ids": [task.task_id]},
    )

    decision = agent.manager_tick(project.project_id)
    assert decision.action == "no_op"


def test_dataos_agent_runs_pending_specialists_in_parallel_batch(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Run pending specialists",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    store.create_agent_task(project.project_id, role="hypothesis", input_context={})
    critic_task = store.create_agent_task(project.project_id, role="critic", input_context={})
    agent = DataOSAgent(tmp_path, RoleEchoLLM())

    report = agent.run_pending_agent_tasks(
        project.project_id,
        roles=["hypothesis"],
        max_workers=2,
    )
    assert len(report.completed_agent_task_ids) == 1
    assert len(report.proposal_ids) == 1
    assert store.get_agent_task(critic_task.agent_task_id).status == "pending"

    report = agent.run_pending_agent_tasks(project.project_id, max_workers=2)
    assert report.completed_agent_task_ids == [critic_task.agent_task_id]
    proposals = store.list_agent_proposals(project.project_id)
    assert {proposal.proposal_type for proposal in proposals} == {"hypotheses", "critique"}


def test_dataos_agent_can_retry_failed_specialist_tasks_explicitly(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Retry failed specialist work",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    agent_task = store.create_agent_task(
        project.project_id,
        role="hypothesis",
        input_context={},
    )
    bad_agent = DataOSAgent(tmp_path, BadResearchLLM())
    with pytest.raises(ValueError, match="non-empty hypotheses"):
        bad_agent.run_agent_task(agent_task.agent_task_id)

    good_agent = DataOSAgent(tmp_path, RoleEchoLLM())
    report = good_agent.run_pending_agent_tasks(project.project_id, max_workers=1)
    assert not report.completed_agent_task_ids
    assert store.get_agent_task(agent_task.agent_task_id).status == "failed"

    report = good_agent.run_pending_agent_tasks(
        project.project_id,
        max_workers=1,
        retry_failed=True,
    )
    assert report.completed_agent_task_ids == [agent_task.agent_task_id]
    assert store.get_agent_task(agent_task.agent_task_id).status == "completed"


def test_dataos_agent_rejects_invalid_specialist_payload(tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Improve invalid proposal handling",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )
    agent_task = store.create_agent_task(
        project.project_id,
        role="hypothesis",
        input_context={"project_id": project.project_id},
    )
    agent = DataOSAgent(tmp_path, BadResearchLLM())

    with pytest.raises(ValueError, match="non-empty hypotheses"):
        agent.run_agent_task(agent_task.agent_task_id)
    assert store.get_agent_task(agent_task.agent_task_id).status == "failed"
    assert not store.list_agent_proposals(project.project_id)


def test_research_cli_project_task_scheduler_worker_flow(tmp_path, capsys):
    task_template_path = _write_task_template(tmp_path)
    workspace = tmp_path / "workspace"

    cli_main(
        [
            "project",
            "create",
            "--workspace",
            str(workspace),
            "--goal",
            "Improve data-centric SFT",
            "--dataset-path",
            "/tmp/dataset",
            "--task-template",
            str(task_template_path),
            "--budget",
            '{"max_runs": 2}',
        ]
    )
    created = json.loads(capsys.readouterr().out)
    project_id = created["project_id"]

    cli_main(
        [
            "task",
            "submit",
            "--workspace",
            str(workspace),
            "--project-id",
            project_id,
            "--type",
            "experiment_run",
            "--spec",
            '{"stub_result": {"score": 0.6}}',
            "--priority",
            "3",
        ]
    )
    submitted = json.loads(capsys.readouterr().out)
    task_id = submitted["task_id"]

    cli_main(["scheduler", "run-ready", "--workspace", str(workspace)])
    run_ready = json.loads(capsys.readouterr().out)
    assert run_ready["scheduler"]["scheduled_task_ids"] == [task_id]
    assert run_ready["worker_reports"][0]["status"] == "completed"
    assert run_ready["worker_reports"][0]["result"]["score"] == 0.6

    cli_main(["project", "compare", "--workspace", str(workspace), project_id])
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["best_run"]["task_id"] == task_id

    cli_main(["task", "get", "--workspace", str(workspace), task_id])
    task_record = json.loads(capsys.readouterr().out)
    assert task_record["task_id"] == task_id
    assert task_record["status"] == "completed"

    cli_main(
        [
            "task",
            "submit",
            "--workspace",
            str(workspace),
            "--project-id",
            project_id,
            "--type",
            "experiment_run",
            "--spec",
            '{"stub_result": {"score": 0.1}}',
        ]
    )
    cancellable = json.loads(capsys.readouterr().out)
    cli_main(
        [
            "task",
            "cancel",
            "--workspace",
            str(workspace),
            cancellable["task_id"],
            "--reason",
            "No longer needed.",
        ]
    )
    cancelled = json.loads(capsys.readouterr().out)
    assert cancelled["status"] == "cancelled"
    assert cancelled["error"] == "No longer needed."
    cli_main(
        [
            "task",
            "requeue",
            "--workspace",
            str(workspace),
            cancellable["task_id"],
            "--reason",
            "Run it after all.",
        ]
    )
    requeued = json.loads(capsys.readouterr().out)
    assert requeued["status"] == "pending"
    assert requeued["error"] == "Run it after all."

    cli_main(
        [
            "event",
            "list",
            "--workspace",
            str(workspace),
            "--project-id",
            project_id,
            "--action",
            "research_task_completed",
        ]
    )
    events = json.loads(capsys.readouterr().out)
    assert len(events) == 1
    assert events[0]["subject_id"] == task_id


def test_research_cli_agent_manager_tick_without_llm_env(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("STARTEAM_BASE_URL", raising=False)
    monkeypatch.delenv("STARTEAM_API_KEY", raising=False)
    monkeypatch.delenv("STARTEAM_MODEL", raising=False)
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Run manager without credentials",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )

    cli_main(
        [
            "agent",
            "manager-tick",
            "--workspace",
            str(tmp_path),
            project.project_id,
        ]
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["action"] == "spawned_initial_agent_tasks"
    assert len(decision["agent_task_ids"]) == 4


def test_research_cli_project_set_status(tmp_path, capsys):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Set project status from CLI",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 1},
    )

    cli_main(
        [
            "project",
            "set-status",
            "--workspace",
            str(tmp_path),
            project.project_id,
            "--status",
            "paused",
        ]
    )
    updated = json.loads(capsys.readouterr().out)
    assert updated["status"] == "paused"

    cli_main(["project", "state", "--workspace", str(tmp_path), project.project_id])
    state = json.loads(capsys.readouterr().out)
    assert state["project"]["status"] == "paused"


def test_research_cli_agent_cycle_runs_parallel_specialists(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(dataos_cli, "_build_llm_client_from_env", lambda: FakeResearchLLM())
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Run agent cycle from CLI",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )

    dataos_cli.main(
        [
            "agent",
            "cycle",
            "--workspace",
            str(tmp_path),
            "--project-id",
            project.project_id,
            "--max-workers",
            "4",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["first_decision"]["action"] == "spawned_initial_agent_tasks"
    assert len(report["agent_run"]["completed_agent_task_ids"]) == 4
    assert report["final_decision"]["action"] == "accepted_proposals"


def test_research_cli_manual_agent_proposal_review(tmp_path, capsys):
    store = ResearchStore(tmp_path)
    project = store.create_project(
        goal="Manually review proposals",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget={"max_runs": 2},
    )
    agent_task = store.create_agent_task(
        project.project_id,
        role="experiment_planner",
        input_context={"project_id": project.project_id},
    )
    critic_agent_task = store.create_agent_task(
        project.project_id,
        role="critic",
        input_context={"project_id": project.project_id},
    )
    accepted_proposal = store.create_agent_proposal(
        project_id=project.project_id,
        agent_task_id=agent_task.agent_task_id,
        role="experiment_planner",
        proposal_type="research_tasks",
        payload={
            "tasks": [
                {
                    "task_type": "experiment_run",
                    "spec": {"stub_result": {"score": 0.7}},
                    "priority": 4,
                }
            ]
        },
    )
    rejected_proposal = store.create_agent_proposal(
        project_id=project.project_id,
        agent_task_id=critic_agent_task.agent_task_id,
        role="critic",
        proposal_type="critique",
        payload={"issues": ["missing ablation"]},
    )

    cli_main(
        [
            "agent",
            "proposal-list",
            "--workspace",
            str(tmp_path),
            "--project-id",
            project.project_id,
            "--status",
            "proposed",
        ]
    )
    proposals = json.loads(capsys.readouterr().out)
    assert {proposal["proposal_id"] for proposal in proposals} == {
        accepted_proposal.proposal_id,
        rejected_proposal.proposal_id,
    }

    cli_main(
        [
            "agent",
            "task-get",
            "--workspace",
            str(tmp_path),
            agent_task.agent_task_id,
        ]
    )
    agent_task_record = json.loads(capsys.readouterr().out)
    assert agent_task_record["agent_task_id"] == agent_task.agent_task_id

    cli_main(
        [
            "agent",
            "proposal-get",
            "--workspace",
            str(tmp_path),
            accepted_proposal.proposal_id,
        ]
    )
    proposal_record = json.loads(capsys.readouterr().out)
    assert proposal_record["proposal_id"] == accepted_proposal.proposal_id

    cli_main(
        [
            "agent",
            "proposal-accept",
            "--workspace",
            str(tmp_path),
            accepted_proposal.proposal_id,
        ]
    )
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["proposal"]["status"] == "accepted"
    assert len(accepted["research_task_ids"]) == 1

    cli_main(
        [
            "agent",
            "proposal-reject",
            "--workspace",
            str(tmp_path),
            rejected_proposal.proposal_id,
            "--reason",
            "Needs more evidence.",
        ]
    )
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["proposal"]["status"] == "rejected"
    assert rejected["proposal"]["reason"] == "Needs more evidence."
    assert len(store.list_research_tasks(project.project_id)) == 1
