from __future__ import annotations

import json

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, EvalArtifact, ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.eval.postprocessors.normalize_latex_answer import extract_normalized_aime_answer
from dataos.eval.types import ModelResponse


def _task(eval_config: dict) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "task-aime2025-eval",
            "name": "aime2025-eval",
            "goal": "run aime2025 eval",
            "input_description": "hf dataset directory",
            "output_description": "artifacts",
            "constraints": {"dataset_format": "hf_datasets"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["eval_model"],
            "train_config": {"mode": "stub"},
            "eval_config": eval_config,
        }
    )


def _dataset_artifact(sandbox: Sandbox, dataset_obj: Dataset | DatasetDict) -> DatasetArtifact:
    artifact = DatasetArtifact(
        artifact_id="dataset-aime2025-1",
        name="dataset",
        path=sandbox.artifact_dir("dataset", "dataset-aime2025-1"),
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "model-aime2025-1")
    return ModelArtifact(
        artifact_id="model-aime2025-1",
        name="model",
        path=model_dir,
        producer="test",
        meta={"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "mode": "base_model_eval"},
    )


def _eval_artifact(artifacts) -> EvalArtifact:
    for artifact in artifacts:
        if isinstance(artifact, EvalArtifact):
            return artifact
    raise AssertionError("EvalArtifact not found")


def test_aime2025_extracts_and_scores_three_digit_integer_answers(tmp_path, monkeypatch):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "question": [
                            "What is the answer to problem 1?",
                            "What is the answer to problem 2?",
                            "What is the answer to problem 3?",
                            "What is the answer to problem 4?",
                        ],
                        "answer": ["42", "7", "999", "4"],
                    }
                )
            }
        ),
    )
    model_artifact = _model_artifact(sandbox)
    task = _task(
        {
            "provider": "native",
            "task": {"name": "aime2025", "config": {}},
            "data": {"split": "test"},
            "adapter": {"name": "transformers", "config": {}},
            "request": {"type": "generate_until", "max_new_tokens": 32},
            "artifacts": {"persist_samples": True},
        }
    )

    def _fake_generate(self, model_artifact, requests, config):
        del self, model_artifact, config
        texts = ["work\n#### 42", r"The final answer is \boxed{007}.", "999", r"\boxed{1000}"]
        return [
            ModelResponse(
                sample_id=request.sample_id,
                index=request.index,
                request_type=request.request_type,
                text=text,
            )
            for request, text in zip(requests, texts)
        ]

    monkeypatch.setattr(
        "dataos.eval.adapters.transformers.TransformersAdapter.generate_until",
        _fake_generate,
    )

    artifacts = run_eval_stage([dataset_artifact, model_artifact], sandbox=sandbox, task=task)
    eval_artifact = _eval_artifact(artifacts)

    assert eval_artifact.meta["task"] == "aime2025"
    assert eval_artifact.meta["score"] == 0.75
    assert eval_artifact.meta["correct"] == 3
    assert eval_artifact.meta["num_examples"] == 4
    assert eval_artifact.meta["invalid_predictions"] == 1

    sample_lines = (eval_artifact.path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in sample_lines]
    assert [sample["predicted_answer"] for sample in samples] == ["042", "007", "999", None]
    assert [sample["reference_answer"] for sample in samples] == ["042", "007", "999", "004"]


def test_aime2025_rejects_non_aime_integer_answers():
    assert extract_normalized_aime_answer("#### 42") == "042"
    assert extract_normalized_aime_answer(r"\boxed{999}") == "999"
    assert extract_normalized_aime_answer("1000") is None
    assert extract_normalized_aime_answer("42.0") is None
