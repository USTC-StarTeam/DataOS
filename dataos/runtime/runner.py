from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataos.core.agent import LLMPlanningAgent, build_agent_from_env
from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, ExperimentArtifact, make_artifact_id, select_working_dataset
from dataos.core.experiment import Experiment
from dataos.core.sandbox import Sandbox
from dataos.core.store import LocalArtifactStore
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.operators import build_default_registry
from dataos.operators.mix import MergeDatasetsOperator
from dataos.train import run_train_stage


@dataclass(slots=True)
class RunTaskResult:
    task: TaskTemplate
    experiment: Experiment
    review: dict[str, Any]
    experiment_artifact: ExperimentArtifact
    initial_artifacts: list[DatasetArtifact]
    agent: LLMPlanningAgent


def run_task(
    task_template: TaskTemplate | str | Path | dict[str, Any],
    datasets: list[str | Path] | str | Path,
    workspace_dir: str | Path = "workspace",
    sandbox_root: str | Path | None = None,
    history_summaries: list[dict[str, Any]] | None = None,
    agent_provider: str | None = None,
) -> RunTaskResult:
    store = LocalArtifactStore(workspace_dir)
    store.prepare()

    task = _load_task(task_template)
    exp_id = make_artifact_id("exp")
    sandbox_path = Path(sandbox_root) if sandbox_root is not None else Path(workspace_dir) / exp_id
    sandbox = Sandbox(sandbox_path)
    sandbox.prepare()

    dataset_sources = _normalize_dataset_inputs(datasets)
    initial_artifacts = _materialize_initial_datasets(dataset_sources, sandbox)

    registry = build_default_registry()
    planning_agent = build_agent_from_env(
        provider=agent_provider,
        workspace=str(sandbox.tmp_dir / "agent"),
    )
    agent = LLMPlanningAgent(agent=planning_agent, registry=registry)

    experiment = Experiment(exp_id=exp_id, task=task, sandbox=sandbox)
    for artifact in initial_artifacts:
        experiment.add_artifact(artifact)
    experiment.start()
    working_artifacts = _build_working_artifacts(initial_artifacts, experiment)

    agent.run_experiment(
        task=task,
        experiment=experiment,
        initial_artifacts=working_artifacts,
        history_summaries=history_summaries,
    )

    _run_training_stage(task=task, experiment=experiment)
    _run_eval_stage(task=task, experiment=experiment)
    review = agent.review(task, experiment)
    experiment_artifact = experiment.finalize()
    store.save(experiment_artifact)

    return RunTaskResult(
        task=task,
        experiment=experiment,
        review=review,
        experiment_artifact=experiment_artifact,
        initial_artifacts=initial_artifacts,
        agent=agent,
    )


def _load_task(task_template: TaskTemplate | str | Path | dict[str, Any]) -> TaskTemplate:
    if isinstance(task_template, TaskTemplate):
        return task_template
    if isinstance(task_template, dict):
        return TaskTemplate.from_dict(task_template)
    return TaskTemplate.from_json(task_template)


def _normalize_dataset_inputs(datasets: list[str | Path] | str | Path) -> list[str]:
    if isinstance(datasets, (str, Path)):
        items = [datasets]
    else:
        items = list(datasets)
    normalized = [str(item).strip() for item in items if str(item).strip()]
    if not normalized:
        raise ValueError("run_task requires at least one dataset input")
    return normalized


def _materialize_initial_datasets(dataset_sources: list[str], sandbox: Sandbox) -> list[DatasetArtifact]:
    artifacts: list[DatasetArtifact] = []
    for index, source in enumerate(dataset_sources, start=1):
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"initial-dataset-{index}",
            path=sandbox.inputs_dir / artifact_id,
            source=source,
            parent_ids=[],
            producer="run_task",
        )
        artifact.materialize()
        dataset_obj = artifact.load()
        if isinstance(dataset_obj, DatasetDict):
            working_dataset, selected_split, available_splits = select_working_dataset(dataset_obj)
            artifact.meta["source"] = str(source)
            artifact.meta["source_selected_split"] = selected_split
            artifact.meta["source_available_splits"] = available_splits
            artifact.save(working_dataset)
        elif isinstance(dataset_obj, Dataset):
            artifact.meta.setdefault("source", str(source))
            artifact.meta.setdefault("source_selected_split", "train")
            artifact.meta.setdefault("source_available_splits", ["train"])
            artifact.save_meta()
        artifact.save_meta()
        artifacts.append(artifact)
    return artifacts


def _build_working_artifacts(initial_artifacts: list[DatasetArtifact], experiment: Experiment) -> list[DatasetArtifact]:
    operator = MergeDatasetsOperator()
    input_artifact_ids = [artifact.artifact_id for artifact in initial_artifacts]
    try:
        merged_artifact = operator.run(initial_artifacts, sandbox=experiment.sandbox)
        merged_artifact.save_meta()
        experiment.add_artifact(merged_artifact)
        experiment.record_step(
            operator_name=operator.name,
            input_artifact_ids=input_artifact_ids,
            output_artifact_ids=[merged_artifact.artifact_id],
            params={},
            status="completed",
        )
        return [merged_artifact]
    except Exception as exc:
        experiment.record_step(
            operator_name=operator.name,
            input_artifact_ids=input_artifact_ids,
            output_artifact_ids=[],
            params={},
            status="failed",
            error=str(exc),
        )
        experiment.fail(exc)
        raise


def _run_training_stage(task: TaskTemplate, experiment: Experiment) -> None:
    _run_runtime_stage(
        experiment=experiment,
        step_name="train_llamafactory",
        params=dict(task.train_config),
        runner=lambda artifacts: run_train_stage(
            artifacts,
            sandbox=experiment.sandbox,
            task=task,
            experiment=experiment,
        ),
    )


def _run_eval_stage(task: TaskTemplate, experiment: Experiment) -> None:
    _run_runtime_stage(
        experiment=experiment,
        step_name="eval_model",
        params=dict(task.eval_config),
        runner=lambda artifacts: run_eval_stage(
            artifacts,
            sandbox=experiment.sandbox,
            task=task,
            experiment=experiment,
        ),
    )


def _run_runtime_stage(
    experiment: Experiment,
    step_name: str,
    params: dict[str, Any],
    runner,
) -> None:
    input_artifact_ids = [artifact.artifact_id for artifact in experiment.artifacts]
    try:
        result = runner(list(experiment.artifacts))
        output_artifacts = result if isinstance(result, list) else [result]
        output_ids: list[str] = []
        for artifact in output_artifacts:
            artifact.save_meta()
            experiment.add_artifact(artifact)
            output_ids.append(artifact.artifact_id)
        experiment.record_step(
            operator_name=step_name,
            input_artifact_ids=input_artifact_ids,
            output_artifact_ids=output_ids,
            params=params,
            status="completed",
            step_type="runtime",
        )
    except Exception as exc:
        experiment.record_step(
            operator_name=step_name,
            input_artifact_ids=input_artifact_ids,
            output_artifact_ids=[],
            params=params,
            status="failed",
            error=str(exc),
            step_type="runtime",
        )
        experiment.fail(exc)
        raise
