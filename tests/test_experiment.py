from __future__ import annotations

import json

from datasets import Dataset

from dataos.core.agent import LLMPlanningAgent
from dataos.core.artifact import DatasetArtifact, EvalArtifact
from dataos.core.experiment import Experiment
from dataos.core.registry import OperatorRegistry
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate


class FakeLLMClient:
    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
        del system_prompt, user_prompt, temperature
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _task() -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "task-1",
            "name": "demo",
            "goal": "run the flow",
            "input_description": "hf dataset directory",
            "output_description": "artifacts",
            "constraints": {"dataset_format": "hf_datasets"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["clean_data", "synthesize_data"],
            "train_config": {"mode": "stub"},
            "eval_config": {"provider": "stub"},
        }
    )


def test_experiment_finalize_writes_report_and_artifact(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    task = _task()
    experiment = Experiment(exp_id="exp-1", task=task, sandbox=sandbox)
    experiment.start()

    dataset_artifact = DatasetArtifact(
        artifact_id="dataset-1",
        name="dataset",
        path=sandbox.artifact_dir("dataset", "dataset-1"),
        producer="test",
    )
    dataset_artifact.save(Dataset.from_dict({"instruction": ["a"], "response": ["b"]}))
    eval_artifact = EvalArtifact(
        artifact_id="eval-1",
        name="eval",
        path=sandbox.artifact_dir("eval", "eval-1"),
        producer="test",
        parent_ids=["dataset-1"],
        meta={"score": 0.75},
    )
    eval_artifact.save_meta()
    experiment.add_artifact(dataset_artifact)
    experiment.add_artifact(eval_artifact)
    experiment.record_step("eval_model", ["dataset-1"], ["eval-1"], {}, "completed")

    experiment_artifact = experiment.finalize()
    report = json.loads((sandbox.root / "report.json").read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert experiment_artifact.meta["score"] == 0.75
    assert (experiment_artifact.path / "report.json").exists()


def test_agent_falls_back_when_plan_json_is_invalid():
    registry = OperatorRegistry()
    for operator_name in ["clean_data", "synthesize_data"]:
        registry._operators[operator_name] = type("Op", (), {"name": operator_name, "describe": lambda self, name=operator_name: {"name": name, "description": name}})()
    agent = LLMPlanningAgent(llm_client=FakeLLMClient({"steps": [{"operator_name": "not_registered", "params": {}}]}), registry=registry)

    steps = agent.plan(
        task=_task(),
        initial_artifact_summaries=[],
        history_summaries=[],
        operator_descriptions=registry.describe_all(),
    )

    assert [step.operator_name for step in steps] == ["clean_data", "synthesize_data"]
