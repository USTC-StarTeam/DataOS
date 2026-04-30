from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

from dataos.core.agent import LLMPlanningAgent
from dataos.core.artifact import DatasetArtifact, ExperimentArtifact, make_artifact_id
from dataos.core.experiment import Experiment
from dataos.core.registry import OperatorRegistry
from dataos.core.sandbox import Sandbox
from dataos.core.store import LocalArtifactStore
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.llm.client import LLMClient
from dataos.llm.openai_compat import (
    DASHSCOPE_BASE_URL,
    DEFAULT_MODEL,
    OpenAICompatLLMClient,
)
from dataos.operators.clean_data import CleanDataOperator
from dataos.operators.synthesize_data import SynthesizeDataOperator
from dataos.train import run_train_stage


@dataclass(slots=True)
class RunTaskResult:
    task: TaskTemplate
    experiment: Experiment
    review: dict[str, Any]
    experiment_artifact: ExperimentArtifact
    archived_experiment_artifact: ExperimentArtifact
    initial_artifact: DatasetArtifact
    store: LocalArtifactStore
    agent: LLMPlanningAgent


def build_default_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    registry.register(CleanDataOperator())
    registry.register(SynthesizeDataOperator())
    return registry


def run_task(
    task_template: TaskTemplate | str | Path | dict[str, Any],
    dataset_path: str | Path,
    workspace_dir: str | Path = "workspace",
    llm_client: LLMClient | None = None,
    sandbox_root: str | Path | None = None,
    history_summaries: list[dict[str, Any]] | None = None,
) -> RunTaskResult:
    store = LocalArtifactStore(workspace_dir)
    store.prepare()

    task = _load_task(task_template)
    exp_id = make_artifact_id("exp")
    sandbox_path = Path(sandbox_root) if sandbox_root is not None else Path(workspace_dir) / "sandboxes" / exp_id
    sandbox = Sandbox(sandbox_path)
    sandbox.prepare()

    initial_artifact = _materialize_initial_dataset(dataset_path, sandbox)
    initial_artifact.save_meta()

    registry = build_default_registry()
    llm = llm_client or _build_client_from_env()
    agent = LLMPlanningAgent(llm_client=llm, registry=registry)

    experiment = Experiment(exp_id=exp_id, task=task, sandbox=sandbox)
    experiment.add_artifact(initial_artifact)
    agent.run_experiment(
        task=task,
        experiment=experiment,
        initial_artifacts=[initial_artifact],
        history_summaries=history_summaries,
    )
    _run_training_stage(task=task, experiment=experiment)
    _run_eval_stage(task=task, experiment=experiment)
    experiment_artifact = experiment.finalize()
    review = agent.review(task, experiment)
    archived_experiment = store.save(experiment_artifact)

    return RunTaskResult(
        task=task,
        experiment=experiment,
        review=review,
        experiment_artifact=experiment_artifact,
        archived_experiment_artifact=archived_experiment,
        initial_artifact=initial_artifact,
        store=store,
        agent=agent,
    )


def _load_task(task_template: TaskTemplate | str | Path | dict[str, Any]) -> TaskTemplate:
    if isinstance(task_template, TaskTemplate):
        return task_template
    if isinstance(task_template, dict):
        return TaskTemplate.from_dict(task_template)
    return TaskTemplate.from_json(task_template)


def _materialize_initial_dataset(dataset_path: str | Path, sandbox: Sandbox) -> DatasetArtifact:
    source = Path(dataset_path)
    if not source.exists():
        raise FileNotFoundError(f"Initial dataset path does not exist: {source}")
    artifact_id = make_artifact_id("dataset")
    target = sandbox.inputs_dir / artifact_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    artifact = DatasetArtifact(
        artifact_id=artifact_id,
        name="initial-dataset",
        path=target,
        parent_ids=[],
        producer="run_task",
    )
    artifact.refresh_metadata_from_disk()
    return artifact


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


def _build_client_from_env() -> LLMClient:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("STARTEAM_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing LLM API key. "
            "Set DASHSCOPE_API_KEY (or legacy STARTEAM_API_KEY); "
            "optional DASHSCOPE_BASE_URL / DASHSCOPE_MODEL override the defaults."
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
