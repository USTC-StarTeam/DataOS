from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dataos.agent import (
    Action,
    Daemon,
    DaemonCadences,
    SpecialistPool,
    StateSnapshot,
    build_snapshot,
)
from dataos.agent.loops import ExecutorLoop, GeneratorLoop, LoopContext
from dataos.agent.policies import analyzer as analyzer_policy
from dataos.agent.policies import executor as executor_policy
from dataos.agent.policies import generator as generator_policy
from dataos.agent.policies import reporter as reporter_policy
from dataos.research.agent import DataOSAgent
from dataos.research.store import ResearchStore


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _CycleLLM:
    """Deterministic LLM for daemon tests — covers the W1 specialist roles."""

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0):
        del user_prompt, temperature
        if "hypothesis" in system_prompt:
            return {
                "proposal_type": "hypotheses",
                "payload": {"hypotheses": [{"title": "Echo hypothesis"}]},
            }
        if "experiment_planner" in system_prompt:
            return {
                "proposal_type": "research_tasks",
                "payload": {
                    "tasks": [
                        {
                            "task_type": "experiment_run",
                            "spec": {"stub_result": {"score": 0.5}},
                        }
                    ]
                },
            }
        if "operator_designer" in system_prompt:
            return {
                "proposal_type": "candidate_operator",
                "payload": {
                    "name": "candidate_filter",
                    "spec": {"kind": "filter"},
                },
            }
        if "critic" in system_prompt:
            return {
                "proposal_type": "critique",
                "payload": {"issues": [], "recommendation": "Proceed."},
            }
        if "evaluation_analyst" in system_prompt:
            return {"proposal_type": "analysis", "payload": {"summary": "ok"}}
        if "report" in system_prompt:
            return {
                "proposal_type": "report",
                "payload": {"summary": {"finding": "done"}},
            }
        raise AssertionError(system_prompt)


def _task_template() -> dict[str, object]:
    return {
        "task_id": "research-test",
        "name": "Research Test",
        "goal": "Improve data-centric SFT",
        "input_description": "HF dataset",
        "output_description": "artifacts",
        "constraints": {"dataset_format": "hf_datasets"},
        "budget": {"max_experiments": 2},
        "max_rounds": 1,
        "pipeline_hint": ["clean_data"],
        "train_config": {"mode": "stub"},
        "eval_config": {"provider": "stub", "provider_config": {"score": 0.5}},
    }


def _make_project(workspace: Path, *, budget: dict[str, object] | None = None) -> str:
    store = ResearchStore(workspace)
    project = store.create_project(
        goal="Daemon test",
        dataset_path="/tmp/dataset",
        task_template=_task_template(),
        budget=budget or {"max_runs": 4},
    )
    return project.project_id


def _snapshot(
    *,
    project_id: str = "proj-1",
    project_status: str = "running",
    research_tasks: tuple[dict, ...] = (),
    agent_tasks: tuple[dict, ...] = (),
    proposed_proposals: tuple[dict, ...] = (),
    accepted_proposals: tuple[dict, ...] = (),
    all_proposals: tuple[dict, ...] | None = None,
    hypotheses: tuple[dict, ...] = (),
    candidate_operators: tuple[dict, ...] = (),
    recent_events: tuple[dict, ...] = (),
    experiment_run_budget: dict | None = None,
    project_concurrency: dict | None = None,
    queue_depth: int = 0,
    wall_clock_hours_limit: float | None = None,
) -> StateSnapshot:
    if all_proposals is None:
        all_proposals = tuple(
            list(proposed_proposals) + list(accepted_proposals)
        )
    return StateSnapshot(
        now=datetime(2026, 4, 30, tzinfo=UTC),
        project={"project_id": project_id, "status": project_status, "budget": {}},
        project_status=project_status,
        research_tasks=research_tasks,
        agent_tasks=agent_tasks,
        proposed_proposals=proposed_proposals,
        accepted_proposals=accepted_proposals,
        all_proposals=all_proposals,
        hypotheses=hypotheses,
        candidate_operators=candidate_operators,
        recent_events=recent_events,
        experiment_run_budget=experiment_run_budget or {"limit": None, "used": 0, "remaining": None},
        project_concurrency=project_concurrency or {"limit": None, "active": 0, "remaining": None},
        queue_depth=queue_depth,
        wall_clock_hours_limit=wall_clock_hours_limit,
    )


# ---------------------------------------------------------------------------
# Action contract
# ---------------------------------------------------------------------------


def test_action_dataclass_is_immutable():
    a = Action(kind="skip", project_id="p1", reason="x")
    with pytest.raises(Exception):
        a.kind = "spawn_specialists"  # type: ignore[misc]
    assert a.to_record()["kind"] == "skip"


# ---------------------------------------------------------------------------
# Generator policy
# ---------------------------------------------------------------------------


def test_generator_decide_initial_campaign_when_empty():
    actions = generator_policy.decide(_snapshot())
    assert len(actions) == 1
    a = actions[0]
    assert a.kind == "spawn_specialists"
    assert set(a.roles) == {"hypothesis", "experiment_planner", "operator_designer", "critic"}
    assert a.context_scope == "initial_campaign"


def test_generator_decide_accepts_proposed_proposals():
    snap = _snapshot(
        proposed_proposals=({"proposal_id": "prop-1", "status": "proposed"},),
    )
    actions = generator_policy.decide(snap)
    assert len(actions) == 1
    assert actions[0].kind == "accept_proposals"
    assert actions[0].proposal_ids == ("prop-1",)


def test_generator_decide_skips_when_inactive():
    actions = generator_policy.decide(_snapshot(project_status="paused"))
    assert actions == [Action(kind="skip", project_id="proj-1", reason="project_inactive")]


def test_generator_decide_idle_when_steady():
    snap = _snapshot(
        agent_tasks=({"agent_task_id": "t1", "status": "running", "role": "critic"},),
    )
    actions = generator_policy.decide(snap)
    assert len(actions) == 1
    assert actions[0].kind == "skip"
    assert actions[0].reason == "generator_idle"


# ---------------------------------------------------------------------------
# Executor policy
# ---------------------------------------------------------------------------


def test_executor_decide_always_emits_tick_when_active():
    actions = executor_policy.decide(_snapshot())
    assert len(actions) == 1
    assert actions[0].kind == "tick_scheduler"


def test_executor_decide_skips_when_inactive():
    actions = executor_policy.decide(_snapshot(project_status="paused"))
    assert actions[0].kind == "skip"


# ---------------------------------------------------------------------------
# Analyzer policy
# ---------------------------------------------------------------------------


def test_analyzer_decide_spawns_on_uncovered_completed():
    snap = _snapshot(
        research_tasks=({"task_id": "t1", "status": "completed"},),
    )
    actions = analyzer_policy.decide(snap)
    assert actions[0].kind == "spawn_specialists"
    assert actions[0].context_scope == "result_analysis"
    assert actions[0].priority == 8


def test_analyzer_decide_skips_when_inflight():
    snap = _snapshot(
        research_tasks=({"task_id": "t1", "status": "completed"},),
        agent_tasks=(
            {
                "agent_task_id": "a1",
                "status": "running",
                "role": "evaluation_analyst",
                "input_context": {"context_scope": "result_analysis"},
            },
        ),
    )
    actions = analyzer_policy.decide(snap)
    assert actions[0].kind == "skip"
    assert actions[0].reason == "analyzer_inflight"


def test_analyzer_decide_skips_when_already_covered():
    snap = _snapshot(
        research_tasks=({"task_id": "t1", "status": "completed"},),
        recent_events=(
            {"action": "analysis_recorded", "payload": {"covered_task_ids": ["t1"]}},
        ),
    )
    actions = analyzer_policy.decide(snap)
    assert actions[0].kind == "skip"
    assert actions[0].reason == "analyzer_no_new_results"


# ---------------------------------------------------------------------------
# Reporter policy
# ---------------------------------------------------------------------------


def test_reporter_decide_triggers_on_budget_exhausted():
    snap = _snapshot(
        research_tasks=({"task_id": "t1", "status": "completed"},),
        experiment_run_budget={"limit": 1, "used": 1, "remaining": 0},
    )
    actions = reporter_policy.decide(snap)
    assert actions[0].kind == "spawn_specialists"
    assert actions[0].roles == ("report",)
    assert actions[0].context_scope == "final_report"


def test_reporter_decide_skips_when_report_already_proposed():
    snap = _snapshot(
        research_tasks=(
            {"task_id": "t1", "status": "completed"},
            {"task_id": "t2", "status": "completed"},
        ),
        all_proposals=(
            {"proposal_id": "p", "proposal_type": "report", "status": "accepted"},
        ),
    )
    actions = reporter_policy.decide(snap)
    assert actions[0].kind == "skip"
    assert actions[0].reason == "reporter_already_proposed"


def test_reporter_decide_skips_when_open_tasks_remain():
    snap = _snapshot(
        research_tasks=(
            {"task_id": "t1", "status": "completed"},
            {"task_id": "t2", "status": "completed"},
            {"task_id": "t3", "status": "scheduled"},
        ),
    )
    actions = reporter_policy.decide(snap)
    assert actions[0].kind == "skip"
    assert actions[0].reason == "reporter_open_tasks"


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_build_snapshot_populates_queue_depth(tmp_path):
    project_id = _make_project(tmp_path)
    store = ResearchStore(tmp_path)
    store.submit_research_task(
        project_id, task_type="experiment_run", spec={"stub_result": {"score": 0.1}}
    )
    store.submit_research_task(
        project_id, task_type="experiment_run", spec={"stub_result": {"score": 0.2}}
    )
    snap = build_snapshot(store, project_id, datetime.now(UTC))
    assert snap.queue_depth == 2
    assert snap.project_status in {"created", "running"}


def test_build_snapshot_wall_clock_hours_limit_optional(tmp_path):
    project_id = _make_project(tmp_path)
    store = ResearchStore(tmp_path)
    snap = build_snapshot(store, project_id, datetime.now(UTC))
    assert snap.wall_clock_hours_limit is None


# ---------------------------------------------------------------------------
# SpecialistPool
# ---------------------------------------------------------------------------


def test_pool_submit_specialists_creates_agent_tasks(tmp_path):
    project_id = _make_project(tmp_path)
    store = ResearchStore(tmp_path)
    agent = DataOSAgent(tmp_path, _CycleLLM())
    pool = SpecialistPool(agent, tmp_path, llm_client=_CycleLLM(), max_workers=2)
    try:
        submissions = pool.submit_specialists(
            project_id,
            ("hypothesis", "critic"),
            "initial_campaign",
            priority=3,
        )
        for sub in submissions:
            sub.future.result(timeout=10)
    finally:
        pool.drain()
    tasks = store.list_agent_tasks(project_id)
    assert {task.role for task in tasks} == {"hypothesis", "critic"}
    assert all(task.status == "completed" for task in tasks)
    proposals = store.list_agent_proposals(project_id)
    assert {p.role for p in proposals} == {"hypothesis", "critic"}


# ---------------------------------------------------------------------------
# Loop cadence
# ---------------------------------------------------------------------------


def test_loop_does_not_double_tick_within_cadence(tmp_path):
    project_id = _make_project(tmp_path)
    agent = DataOSAgent(tmp_path)
    pool = SpecialistPool(agent, tmp_path, max_workers=1)
    try:
        ctx = LoopContext(
            store=agent.store,
            agent=agent,
            pool=pool,
            project_id=project_id,
            workspace_dir=tmp_path,
        )
        clock_state = {"now": 100.0}

        def clock() -> float:
            return clock_state["now"]

        loop = ExecutorLoop(ctx, cadence_seconds=10.0, monotonic_clock=clock)
        loop.tick()
        clock_state["now"] = 105.0  # within cadence
        actions = loop.tick()
        assert actions == []  # not due
        clock_state["now"] = 115.0  # past cadence
        actions = loop.tick()
        assert any(a.kind == "tick_scheduler" for a in actions)
    finally:
        pool.drain()


# ---------------------------------------------------------------------------
# Executor loop end-to-end
# ---------------------------------------------------------------------------


def test_executor_loop_calls_scheduler_and_workers(tmp_path):
    project_id = _make_project(tmp_path)
    store = ResearchStore(tmp_path)
    task = store.submit_research_task(
        project_id, task_type="experiment_run", spec={"stub_result": {"score": 0.42}}
    )
    agent = DataOSAgent(tmp_path)
    pool = SpecialistPool(agent, tmp_path, max_workers=2)
    try:
        ctx = LoopContext(
            store=store,
            agent=agent,
            pool=pool,
            project_id=project_id,
            workspace_dir=tmp_path,
        )
        ExecutorLoop(ctx, cadence_seconds=0.0).tick()
    finally:
        pool.drain()
    refreshed = store.get_research_task(task.task_id)
    assert refreshed.status == "completed"
    assert refreshed.result.get("score") == 0.42


# ---------------------------------------------------------------------------
# Daemon lifecycle
# ---------------------------------------------------------------------------


def test_daemon_startup_recovery_runs_before_loop_ticks(tmp_path):
    project_id = _make_project(tmp_path)
    store = ResearchStore(tmp_path)
    store.add_resource(
        resource_id="local-test",
        host="local",
        kind="local",
        gpu_ids=[],
        max_concurrent_tasks=1,
    )
    task = store.submit_research_task(
        project_id, task_type="experiment_run", spec={"stub_result": {"score": 0.1}}
    )
    resource = store.list_resources()[0]
    store.schedule_research_task(task, resource, actor="test")

    daemon = Daemon(
        workspace_dir=tmp_path,
        project_id=project_id,
        cadences=DaemonCadences(generator=60, executor=60, analyzer=60, reporter=60),
        stale_seconds=0,
        install_signal_handlers=False,
    )
    daemon.start()
    daemon.shutdown(drain_timeout=5.0)

    events = store.list_events(project_id=project_id)
    actions = [e.action for e in events]
    assert "recovery_action" in actions
    assert "daemon_started" in actions
    recovery_idx = actions.index("recovery_action")
    started_idx = actions.index("daemon_started")
    assert recovery_idx < started_idx
    recovery_event = events[recovery_idx]
    assert task.task_id in recovery_event.payload.get("requeued_task_ids", [])


def test_daemon_shutdown_drains_within_n_seconds(tmp_path):
    project_id = _make_project(tmp_path)
    daemon = Daemon(
        workspace_dir=tmp_path,
        project_id=project_id,
        cadences=DaemonCadences(generator=60, executor=60, analyzer=60, reporter=60),
        stale_seconds=900,
        install_signal_handlers=False,
    )
    daemon.start()
    time.sleep(0.1)
    threads = list(daemon._threads)
    assert all(t.is_alive() for t in threads)

    started_at = time.monotonic()
    daemon.shutdown(drain_timeout=5.0)
    elapsed = time.monotonic() - started_at

    assert elapsed < 5.0
    assert all(not t.is_alive() for t in threads)

    store = ResearchStore(tmp_path)
    actions = [e.action for e in store.list_events(project_id=project_id)]
    assert "daemon_stopped" in actions
