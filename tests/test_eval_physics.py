from __future__ import annotations

import json

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, EvalArtifact, ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.eval.postprocessors.normalize_latex_answer import extract_normalized_latex_answer
from dataos.eval.types import ModelResponse


def _task(eval_config: dict) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "task-physics-eval",
            "name": "physics-eval",
            "goal": "run physics eval",
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
        artifact_id="dataset-physics-1",
        name="dataset",
        path=sandbox.artifact_dir("dataset", "dataset-physics-1"),
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "model-physics-1")
    return ModelArtifact(
        artifact_id="model-physics-1",
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


def test_physics_normalizes_latex_answers_without_symbolic_equivalence(tmp_path, monkeypatch):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "question": [
                            "How much does the internal energy increase?",
                            "What is the coefficient?",
                            "Give the final value.",
                        ],
                        "answer": [
                            [[r"\boxed{3.75 \times 10^{4}}"]],
                            [[r"\boxed{\frac{1}{2}}"]],
                            [[r"\boxed{3.75 \times 10^{4}}"]],
                        ],
                    }
                )
            }
        ),
    )
    model_artifact = _model_artifact(sandbox)
    task = _task(
        {
            "provider": "native",
            "task": {"name": "physics", "config": {}},
            "data": {"split": "test"},
            "adapter": {"name": "transformers", "config": {}},
            "request": {"type": "generate_until", "max_new_tokens": 64},
            "artifacts": {"persist_samples": True},
        }
    )

    def _fake_generate(self, model_artifact, requests, config):
        del self, model_artifact, config
        texts = [
            r"work #### \boxed{3.75 \times 10^4}",
            r"The final answer is \boxed{\dfrac{1}{2}}.",
            "37500",
        ]
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

    assert eval_artifact.meta["task"] == "physics"
    assert eval_artifact.meta["metric"] == "exact_match"
    assert eval_artifact.meta["matching_policy"] == "normalized_string_exact_match"
    assert eval_artifact.meta["score"] == 2 / 3
    assert eval_artifact.meta["correct"] == 2
    assert eval_artifact.meta["num_examples"] == 3

    sample_lines = (eval_artifact.path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in sample_lines]
    assert [sample["predicted_answer"] for sample in samples] == [
        "3.75*10^4",
        "1/2",
        "37500",
    ]
    assert [sample["reference_answer"] for sample in samples] == [
        "3.75*10^4",
        "1/2",
        "3.75*10^4",
    ]
    assert [sample["is_correct"] for sample in samples] == [True, True, False]


def test_physics_latex_normalization_is_string_based():
    assert extract_normalized_latex_answer(r"#### \boxed{\dfrac{1}{2}}") == "1/2"
    assert extract_normalized_latex_answer("1/2") == "1/2"
    assert extract_normalized_latex_answer(r"\boxed{3.75 \times 10^{4}}") == "3.75*10^4"
    assert extract_normalized_latex_answer("37500") != "3.75*10^4"
    assert extract_normalized_latex_answer("answer unknown") is None
    assert extract_normalized_latex_answer("#### B") == "B"
    assert extract_normalized_latex_answer("#### A, B, D") == "A"


def test_physics_multi_answer_requires_full_normalized_sequence(tmp_path, monkeypatch):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "question": ["Give both final values.", "Give both final values."],
                        "answer": [
                            [[r"\boxed{1}"], [r"\boxed{2}"]],
                            [[r"\boxed{1}"], [r"\boxed{2}"]],
                        ],
                    }
                )
            }
        ),
    )
    model_artifact = _model_artifact(sandbox)
    task = _task(
        {
            "provider": "native",
            "task": {"name": "physics", "config": {}},
            "data": {"split": "test"},
            "adapter": {"name": "transformers", "config": {}},
            "request": {"type": "generate_until", "max_new_tokens": 64},
            "artifacts": {"persist_samples": True},
        }
    )

    def _fake_generate(self, model_artifact, requests, config):
        del self, model_artifact, config
        texts = [r"#### \boxed{1}, \boxed{2}", r"#### \boxed{1}"]
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

    assert eval_artifact.meta["score"] == 0.5
    samples = [
        json.loads(line)
        for line in (eval_artifact.path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [sample["predicted_answer"] for sample in samples] == ["1 ; 2", "1"]
    assert [sample["reference_answer"] for sample in samples] == ["1 ; 2", "1 ; 2"]
    assert [sample["is_correct"] for sample in samples] == [True, False]
