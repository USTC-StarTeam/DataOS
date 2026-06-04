"""Agent-less runtime entrypoint for already-prepared training datasets.

This is the "skill version" runtime: a coding agent (Claude Code / Codex) acts
as the orchestrator/planner. It composes ``dataos-operator`` operators itself to
build a training set, exports an Alpaca/ShareGPT JSONL, and then calls
``run_prepared`` to drive only the deterministic, fixed stages that remain code:
train (stub) and eval (provider=stub), followed by finalize.

Unlike :func:`dataos.runtime.runner.run_task`, this entrypoint does NOT embed an
``LLMPlanningAgent`` and never requires an LLM API key. The data-preparation
pipeline lives outside this module, owned by the calling agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import (
    DatasetArtifact,
    ExperimentArtifact,
    make_artifact_id,
    select_working_dataset,
)
from dataos.core.experiment import Experiment
from dataos.core.sandbox import Sandbox
from dataos.core.store import LocalArtifactStore
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.train import run_train_stage


@dataclass(slots=True)
class RunPreparedResult:
    task: TaskTemplate
    experiment: Experiment
    experiment_artifact: ExperimentArtifact
    dataset_artifact: DatasetArtifact


def run_prepared(
    task_template: TaskTemplate | str | Path | dict[str, Any],
    dataset_path: str | Path,
    workspace_dir: str | Path = "workspace",
    sandbox_root: str | Path | None = None,
) -> RunPreparedResult:
    """Run the fixed train+eval+finalize stages over a prepared training set.

    ``dataset_path`` points at the training data the skill already produced with
    ``dataos-operator`` (typically an Alpaca/ShareGPT ``.jsonl``). It may also be
    a ``.json``/``.csv`` file or a Hugging Face ``save_to_disk`` directory; the
    standard :class:`DatasetArtifact` loaders handle each case.

    The runtime materializes that dataset as the single working
    :class:`DatasetArtifact`, runs ``run_train_stage`` (stub) and then
    ``run_eval_stage`` (provider=stub), finalizes the experiment (writing
    ``report.json``), and persists an :class:`ExperimentArtifact`.
    """
    store = LocalArtifactStore(workspace_dir)
    store.prepare()

    task = _load_task(task_template)
    exp_id = make_artifact_id("exp")
    sandbox_path = Path(sandbox_root) if sandbox_root is not None else Path(workspace_dir) / exp_id
    sandbox = Sandbox(sandbox_path)
    sandbox.prepare()

    dataset_artifact = _materialize_prepared_dataset(dataset_path, sandbox)

    experiment = Experiment(exp_id=exp_id, task=task, sandbox=sandbox)
    experiment.add_artifact(dataset_artifact)
    experiment.start()

    _run_train_stage(task=task, experiment=experiment)
    _run_eval_stage(task=task, experiment=experiment)

    experiment_artifact = experiment.finalize()
    store.save(experiment_artifact)

    return RunPreparedResult(
        task=task,
        experiment=experiment,
        experiment_artifact=experiment_artifact,
        dataset_artifact=dataset_artifact,
    )


def _load_task(task_template: TaskTemplate | str | Path | dict[str, Any]) -> TaskTemplate:
    if isinstance(task_template, TaskTemplate):
        return task_template
    if isinstance(task_template, dict):
        return TaskTemplate.from_dict(task_template)
    return TaskTemplate.from_json(task_template)


def _materialize_prepared_dataset(dataset_path: str | Path, sandbox: Sandbox) -> DatasetArtifact:
    source = str(dataset_path).strip()
    if not source:
        raise ValueError("run_prepared requires a non-empty dataset_path")
    if not Path(source).expanduser().exists():
        raise FileNotFoundError(f"Prepared dataset path does not exist: {source}")

    artifact_id = make_artifact_id("dataset")
    artifact = DatasetArtifact(
        artifact_id=artifact_id,
        name="prepared-training-dataset",
        path=sandbox.inputs_dir / artifact_id,
        source=source,
        parent_ids=[],
        producer="run_prepared",
    )
    artifact.materialize()
    dataset_obj = artifact.load()
    if isinstance(dataset_obj, DatasetDict):
        working_dataset, selected_split, available_splits = select_working_dataset(dataset_obj)
        artifact.meta["source"] = source
        artifact.meta["source_selected_split"] = selected_split
        artifact.meta["source_available_splits"] = available_splits
        artifact.save(working_dataset)
    elif isinstance(dataset_obj, Dataset):
        artifact.meta.setdefault("source", source)
        artifact.meta.setdefault("source_selected_split", "train")
        artifact.meta.setdefault("source_available_splits", ["train"])
    artifact.save_meta()
    return artifact


def _run_train_stage(task: TaskTemplate, experiment: Experiment) -> None:
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
