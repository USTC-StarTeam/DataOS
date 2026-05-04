"""Integration test: run_eval_stage drives lm-eval and emits DataOS artifacts.

Uses a fake adapter (registered in-place) so we exercise the full runner path
— config resolution, adapter dispatch, artifact persistence — without GPU or
real model weights. lm-eval is required; gsm8k is fetched from
``HF_ENDPOINT=https://hf-mirror.com`` if the mirror is set.
"""
from __future__ import annotations

import json
import os

import pytest

from dataos.core.artifact import EvalArtifact, ModelArtifact, ResultArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.eval.types import EvalConfig, EvalRequest, ModelResponse

pytest.importorskip("lm_eval")
pytest.skip("phase B reverted: relies on new EvalConfig schema (tasks/adapter)", allow_module_level=True)


class _FakeAdapter:
    name = "fake"
    instances: list["_FakeAdapter"] = []

    def __init__(self):
        self.calls = 0
        type(self).instances.append(self)

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: EvalConfig,
    ) -> list[ModelResponse]:
        self.calls += 1
        return [
            ModelResponse(
                sample_id=request.sample_id,
                index=request.index,
                request_type="generate_until",
                text="answer is 18 #### 18",
            )
            for request in requests
        ]


def _task() -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "task-eval-runner",
            "name": "eval-runner",
            "goal": "run lm-eval gsm8k",
            "input_description": "model artifact",
            "output_description": "eval artifact",
            "constraints": {"dataset_format": "hf_datasets"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["eval_model"],
            "train_config": {"mode": "stub"},
            "eval_config": {
                "tasks": ["gsm8k"],
                "adapter": {"name": "fake"},
                "limit": 3,
                "num_fewshot": 0,
                "request": {"batch_size": 3, "max_new_tokens": 16},
                "apply_chat_template": False,
            },
        }
    )


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "fake-model-runner")
    model_dir.mkdir(parents=True, exist_ok=True)
    return ModelArtifact(
        artifact_id="fake-model-runner",
        name="fake",
        path=model_dir,
        producer="test",
        meta={"base_model": "fake/qwen-stub"},
    )


def test_run_eval_stage_persists_eval_and_sample_artifacts(tmp_path, monkeypatch):
    if not os.environ.get("HF_ENDPOINT") and not os.environ.get("HF_DATASETS_OFFLINE"):
        pytest.skip("HF_ENDPOINT not set; avoid network reach to huggingface.co")

    monkeypatch.setitem(
        __import__("dataos.eval.runner", fromlist=["ADAPTERS"]).ADAPTERS,
        "fake",
        _FakeAdapter,
    )
    _FakeAdapter.instances.clear()

    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    model = _model_artifact(sandbox)
    task = _task()

    artifacts = run_eval_stage([model], sandbox=sandbox, task=task)

    eval_artifacts = [a for a in artifacts if isinstance(a, EvalArtifact)]
    sample_artifacts = [a for a in artifacts if isinstance(a, ResultArtifact)]
    assert len(eval_artifacts) == 1
    eval_artifact = eval_artifacts[0]
    assert len(sample_artifacts) == 1
    sample_artifact = sample_artifacts[0]

    assert eval_artifact.meta["backend"] == "lm_eval"
    assert eval_artifact.meta["tasks"] == ["gsm8k"]
    assert eval_artifact.meta["adapter"] == "fake"
    assert eval_artifact.meta["limit"] == 3
    assert eval_artifact.meta["score"] is not None
    assert "gsm8k" in eval_artifact.meta["results"]
    assert eval_artifact.meta["sample_artifact_ids"]["gsm8k"] == sample_artifact.artifact_id

    results_path = eval_artifact.path / "results.json"
    raw_path = eval_artifact.path / "lm_eval_results.json"
    assert results_path.exists()
    assert raw_path.exists()
    summary = json.loads(results_path.read_text(encoding="utf-8"))
    assert summary["backend"] == "lm_eval"

    sample_lines = (sample_artifact.path / "samples.jsonl").read_text(encoding="utf-8").strip().splitlines()
    docs_seen = {json.loads(line)["doc_id"] for line in sample_lines}
    assert docs_seen == {0, 1, 2}

    assert _FakeAdapter.instances and _FakeAdapter.instances[-1].calls >= 1
