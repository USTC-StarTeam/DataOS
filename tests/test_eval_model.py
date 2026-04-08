from __future__ import annotations

import json

import pytest
from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, EvalArtifact, ModelArtifact, ResultArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.eval.adapters.transformers import _render_chat_prompt
from dataos.eval.types import EvalAdapterConfig, EvalConfig, EvalRequestConfig
from dataos.eval.types import ModelResponse


def _task(eval_config: dict) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "task-eval",
            "name": "eval",
            "goal": "run eval",
            "input_description": "hf dataset directory",
            "output_description": "artifacts",
            "constraints": {"dataset_format": "hf_datasets"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["clean_data", "synthesize_data"],
            "train_config": {"mode": "stub"},
            "eval_config": eval_config,
        }
    )


def _dataset_artifact(sandbox: Sandbox, dataset_obj: Dataset | DatasetDict) -> DatasetArtifact:
    artifact = DatasetArtifact(
        artifact_id="dataset-1",
        name="dataset",
        path=sandbox.artifact_dir("dataset", "dataset-1"),
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "model-1")
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    return ModelArtifact(
        artifact_id="model-1",
        name="model",
        path=model_dir,
        producer="test",
        meta={"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "mode": "llamafactory"},
    )


def _eval_artifact(artifacts):
    for artifact in artifacts:
        if isinstance(artifact, EvalArtifact):
            return artifact
    raise AssertionError("EvalArtifact not found")


def _result_artifact(artifacts, kind: str) -> ResultArtifact:
    for artifact in artifacts:
        if isinstance(artifact, ResultArtifact) and artifact.meta.get("kind") == kind:
            return artifact
    raise AssertionError(f"ResultArtifact(kind={kind}) not found")


def test_eval_model_native_v2_scores_predictions_and_persists_traces(tmp_path, monkeypatch):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "instruction": ["train question"],
                        "response": ["work #### 1"],
                    }
                ),
                "test": Dataset.from_dict(
                    {
                        "instruction": ["question 1", "question 2"],
                        "response": ["reasoning #### 5", "reasoning #### 9"],
                        "answer": ["reasoning #### 5", "reasoning #### 9"],
                    }
                ),
            }
        ),
    )
    model_artifact = _model_artifact(sandbox)
    task = _task(
        {
            "provider": "native",
            "task": {"name": "gsm8k", "config": {}},
            "data": {"split": "test"},
            "adapter": {
                "name": "transformers",
                "config": {"trust_remote_code": False},
            },
            "request": {
                "type": "generate_until",
                "batch_size": 2,
                "max_new_tokens": 32,
                "do_sample": False,
            },
            "artifacts": {
                "persist_requests": True,
                "persist_responses": True,
                "persist_samples": True,
            },
        }
    )

    def _fake_generate(self, model_artifact, requests, config):
        del self, model_artifact, config
        texts = ["I think the answer is 5", "Maybe 7"]
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

    artifacts = run_eval_stage(
        [dataset_artifact, model_artifact],
        sandbox=sandbox,
        task=task,
    )

    eval_artifact = _eval_artifact(artifacts)
    request_artifact = _result_artifact(artifacts, "eval_requests")
    response_artifact = _result_artifact(artifacts, "eval_responses")
    sample_artifact = _result_artifact(artifacts, "eval_samples")

    assert eval_artifact.meta["provider"] == "native"
    assert eval_artifact.meta["task"] == "gsm8k"
    assert eval_artifact.meta["adapter"] == "transformers"
    assert eval_artifact.meta["request_type"] == "generate_until"
    assert eval_artifact.meta["metric"] == "exact_match"
    assert eval_artifact.meta["dataset_split"] == "test"
    assert eval_artifact.meta["score"] == 0.5
    assert eval_artifact.meta["correct"] == 1
    assert eval_artifact.meta["num_examples"] == 2
    assert eval_artifact.meta["trace_artifact_ids"] == {
        "requests": request_artifact.artifact_id,
        "responses": response_artifact.artifact_id,
        "samples": sample_artifact.artifact_id,
    }

    summary = json.loads((eval_artifact.path / "results.json").read_text(encoding="utf-8"))
    assert summary["task_version"] == "v1"
    assert summary["trace_artifact_ids"]["samples"] == sample_artifact.artifact_id

    sample_lines = (eval_artifact.path / "samples.jsonl").read_text(encoding="utf-8").strip().splitlines()
    prediction_lines = (eval_artifact.path / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sample_lines) == 2
    assert sample_lines == prediction_lines

    request_trace = (request_artifact.path / "requests.jsonl").read_text(encoding="utf-8").strip().splitlines()
    response_trace = (response_artifact.path / "responses.jsonl").read_text(encoding="utf-8").strip().splitlines()
    sample_trace = (sample_artifact.path / "samples.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(request_trace) == 2
    assert len(response_trace) == 2
    assert len(sample_trace) == 2


def test_eval_model_stub_provider_returns_configured_score(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    model_artifact = _model_artifact(sandbox)
    task = _task({"provider": "stub", "provider_config": {"score": 0.75}})

    artifacts = run_eval_stage([model_artifact], sandbox=sandbox, task=task)
    eval_artifact = _eval_artifact(artifacts)

    assert eval_artifact.meta["provider"] == "stub"
    assert eval_artifact.meta["score"] == 0.75


def test_eval_model_gsm8k_requires_split_to_exist(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "instruction": ["question 1"],
                        "response": ["reasoning #### 5"],
                    }
                )
            }
        ),
    )
    model_artifact = _model_artifact(sandbox)
    task = _task(
        {
            "provider": "native",
            "task": {"name": "gsm8k", "config": {}},
            "data": {"split": "test"},
            "adapter": {"name": "transformers", "config": {}},
            "request": {"type": "generate_until"},
        }
    )

    with pytest.raises(ValueError, match="Requested eval split 'test' not found"):
        run_eval_stage([dataset_artifact, model_artifact], sandbox=sandbox, task=task)


def test_eval_model_gsm8k_rejects_unknown_step_param(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "instruction": ["question 1"],
                        "response": ["reasoning #### 5"],
                    }
                )
            }
        ),
    )
    model_artifact = _model_artifact(sandbox)
    task = _task({"provider": "stub"})

    with pytest.raises(ValueError, match="Unsupported eval_model step params: unknown_flag"):
        run_eval_stage(
            [dataset_artifact, model_artifact],
            sandbox=sandbox,
            task=task,
            unknown_flag=True,
        )


def test_render_chat_prompt_passes_enable_thinking_when_supported():
    class FakeTokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = {"messages": messages, **kwargs}
            return "rendered"

    tokenizer = FakeTokenizer()
    config = EvalConfig(
        provider="native",
        adapter=EvalAdapterConfig(name="transformers", config={"enable_thinking": False}),
        request=EvalRequestConfig(type="generate_until"),
    )

    rendered = _render_chat_prompt(tokenizer, "hello", config)

    assert rendered == "rendered"
    assert tokenizer.kwargs["enable_thinking"] is False
