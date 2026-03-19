from __future__ import annotations

from datetime import datetime

from dataos.core import EvalResult, OptimizationState, PipelineStep, PipelineStepStatus
from dataos.runtime import RuntimeStateStore
from dataos.storage import RunSnapshot, RunStore, default_run_store_path


def test_run_store_persists_and_queries_run_snapshot(test_workspace) -> None:
    store = RunStore(test_workspace / "artifacts" / "runs")
    plan = [
        PipelineStep(
            step_id="clean-1",
            operator_id="filter_by_rule",
            inputs=["dataset:raw:v1"],
            config={"min_score": 0.8},
        ),
        PipelineStep(
            step_id="score-1",
            operator_id="filter_by_score",
            inputs=["dataset:clean:v2"],
            config={"threshold": 0.8},
        ),
    ]

    created = store.create_run(
        "run-42",
        plan=plan,
        metadata={"task_id": "task-sft", "controller": {"round": 1}},
    )

    assert created.run_id == "run-42"
    assert [step.step_id for step in created.plan] == ["clean-1", "score-1"]
    assert created.metrics == {}
    assert created.eval_results == []

    store.save_step(
        "run-42",
        PipelineStep(
            step_id="clean-1",
            operator_id="filter_by_rule",
            inputs=["dataset:raw:v1"],
            outputs=["dataset:clean:v2"],
            config={"min_score": 0.8},
            status=PipelineStepStatus.SUCCEEDED,
            metrics={"rows_in": 100, "rows_out": 87},
        ),
    )
    store.update_metrics(
        "run-42",
        {"runtime": {"seconds": 12.5}, "cost": {"api_usd": 0.14}},
    )
    store.update_metrics(
        "run-42",
        {"cost": {"cpu_seconds": 4.2}, "eval": {"win_rate": 0.81}},
    )
    store.save_eval_result(
        "run-42",
        EvalResult(
            eval_id="eval-20260313",
            dataset_version="dataset:clean:v2",
            task_id="task-sft",
            metrics={"win_rate": 0.81, "format.valid_rate": 0.99},
            artifacts={"report": "artifacts/runs/run-42/eval.json"},
            summary={"winner": "cleaned"},
        ),
    )
    store.save_optimization_state(
        "run-42",
        OptimizationState(
            round_id="round-1",
            candidate_configs=[{"name": "baseline"}, {"name": "cleaned"}],
            tried_configs=[{"name": "cleaned"}],
            best_config={"name": "cleaned"},
            best_result={
                "eval_id": "eval-20260313",
                "dataset_version": "dataset:clean:v2",
                "task_id": "task-sft",
                "metrics": {"win_rate": 0.81},
            },
            history=[{"candidate": "cleaned", "metrics": {"win_rate": 0.81}}],
        ),
    )

    snapshot = store.load_run("run-42")

    assert snapshot is not None
    assert [step.step_id for step in snapshot.steps] == ["clean-1", "score-1"]
    assert snapshot.get_step("clean-1") is not None
    assert snapshot.get_step("clean-1").status == PipelineStepStatus.SUCCEEDED
    assert snapshot.get_step("score-1").status == PipelineStepStatus.PENDING
    assert snapshot.metrics == {
        "runtime": {"seconds": 12.5},
        "cost": {"api_usd": 0.14, "cpu_seconds": 4.2},
        "eval": {"win_rate": 0.81},
    }
    assert snapshot.latest_eval_result is not None
    assert snapshot.latest_eval_result.eval_id == "eval-20260313"
    assert snapshot.optimization_state is not None
    assert snapshot.optimization_state.best_result is not None
    assert snapshot.optimization_state.best_result.metrics["win_rate"] == 0.81
    assert snapshot.metadata == {"task_id": "task-sft", "controller": {"round": 1}}
    assert datetime.fromisoformat(snapshot.created_at.replace("Z", "+00:00"))
    assert datetime.fromisoformat(snapshot.updated_at.replace("Z", "+00:00"))


def test_run_store_supports_incremental_updates_and_run_listing(test_workspace) -> None:
    store = RunStore(test_workspace / "artifacts" / "runs")

    store.create_run("run/alpha", metadata={"owner": "runtime"})
    store.update_metadata("run/alpha", {"controller": {"round": 2}})
    store.save_step(
        "run/alpha",
        {
            "step_id": "score/1",
            "operator_id": "filter_by_score",
            "status": "running",
            "metrics": {"rows_seen": 16},
        },
    )
    store.save_eval_result(
        "run/alpha",
        {
            "eval_id": "eval-a",
            "dataset_version": "dataset:v2",
            "task_id": "task-a",
            "metrics": {"quality.score": 0.7},
        },
    )
    store.save_eval_result(
        "run/alpha",
        {
            "eval_id": "eval-a",
            "dataset_version": "dataset:v2",
            "task_id": "task-a",
            "metrics": {"quality.score": 0.74},
        },
    )

    loaded_step = store.load_step("run/alpha", "score/1")
    loaded_eval = store.load_latest_eval_result("run/alpha")
    snapshot = store.load_run("run/alpha")

    assert loaded_step is not None
    assert loaded_step.step_id == "score/1"
    assert loaded_step.status == PipelineStepStatus.RUNNING
    assert loaded_eval is not None
    assert loaded_eval.metrics == {"quality.score": 0.74}
    assert snapshot is not None
    assert snapshot.metadata == {"owner": "runtime", "controller": {"round": 2}}
    assert store.list_runs() == ["run/alpha"]


def test_run_store_implements_runtime_state_store_roundtrip(test_workspace) -> None:
    store = RunStore(test_workspace / "artifacts" / "runs")
    snapshot = RunSnapshot(
        run_id="run-state-1",
        plan=[PipelineStep(step_id="plan-1", operator_id="utility.prefix")],
        steps=[
            PipelineStep(
                step_id="plan-1",
                operator_id="utility.prefix",
                status=PipelineStepStatus.SUCCEEDED,
                metrics={"rows_out": 3},
            )
        ],
        metrics={"runtime": {"seconds": 1.2}},
        eval_results=[
            EvalResult(
                eval_id="eval-state-1",
                dataset_version="dataset:v1",
                task_id="task-state",
                metrics={"quality.score": 0.9},
            )
        ],
        metadata={"owner": "controller"},
    )

    store.save(snapshot)
    loaded = store.load("run-state-1")

    assert isinstance(store, RuntimeStateStore)
    assert loaded is not None
    assert [step.step_id for step in loaded.plan] == ["plan-1"]
    assert loaded.get_step("plan-1") is not None
    assert loaded.get_step("plan-1").status == PipelineStepStatus.SUCCEEDED
    assert loaded.metrics == {"runtime": {"seconds": 1.2}}
    assert loaded.latest_eval_result is not None
    assert loaded.latest_eval_result.eval_id == "eval-state-1"
    assert loaded.metadata == {"owner": "controller"}


def test_default_run_store_path_uses_artifacts_runs(test_workspace) -> None:
    path = default_run_store_path(str(test_workspace))

    assert path == test_workspace / "artifacts" / "runs"
