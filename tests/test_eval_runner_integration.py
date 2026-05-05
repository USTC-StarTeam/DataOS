"""Integration: ``run_eval_stage`` drives lm-eval and emits EvalArtifact.

Uses a fake adapter monkey-patched into the runner's ``_make_adapter``
factory so the full runtime path (config resolution -> adapter dispatch
-> artifact persistence) is exercised without GPU or real model weights.
``lm_eval`` must be importable; the gsm8k task is fetched from
``HF_ENDPOINT`` (``HF_ENDPOINT=https://hf-mirror.com`` works behind the
GFW).
"""

from __future__ import annotations

import json
import os

import pytest

from dataos.core.artifact import EvalArtifact, ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.eval.types import AdapterCallContext, EvalRequest, ModelResponse

pytest.importorskip("lm_eval")


class _FakeAdapter:
    name = "fake"
    instances: list["_FakeAdapter"] = []

    def __init__(self) -> None:
        self.calls = 0
        type(self).instances.append(self)

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: AdapterCallContext,
    ) -> list[ModelResponse]:
        del model_artifact, config
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
                "limit": 3,
                "num_fewshot": 0,
                "batch_size": 3,
                "max_new_tokens": 16,
                "apply_chat_template": False,
                "adapter": "fake",
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


def test_run_eval_stage_persists_eval_artifact_via_fake_adapter(tmp_path, monkeypatch):
    if not os.environ.get("HF_ENDPOINT") and not os.environ.get("HF_DATASETS_OFFLINE"):
        pytest.skip("HF_ENDPOINT not set; avoid network reach to huggingface.co")

    _FakeAdapter.instances.clear()
    monkeypatch.setattr("dataos.eval.runner._make_adapter", lambda name, config: _FakeAdapter())

    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    model = _model_artifact(sandbox)
    task = _task()

    artifacts = run_eval_stage([model], sandbox=sandbox, task=task)

    eval_artifacts = [artifact for artifact in artifacts if isinstance(artifact, EvalArtifact)]
    assert len(eval_artifacts) == 1
    eval_artifact = eval_artifacts[0]

    assert eval_artifact.meta["backend"] == "lm_eval"
    assert eval_artifact.meta["tasks"] == ["gsm8k"]
    assert eval_artifact.meta["adapter"] == "fake"
    assert eval_artifact.meta["limit"] == 3
    assert "gsm8k" in eval_artifact.meta["results"]
    assert eval_artifact.meta["headline_score"] is not None
    assert eval_artifact.meta["score"] == eval_artifact.meta["headline_score"]

    results_path = eval_artifact.path / "results.json"
    raw_path = eval_artifact.path / "lm_eval_results.json"
    assert results_path.exists()
    assert raw_path.exists()
    summary = json.loads(results_path.read_text(encoding="utf-8"))
    assert summary["backend"] == "lm_eval"

    samples_path = eval_artifact.path / "samples-gsm8k.jsonl"
    assert samples_path.exists()
    docs_seen = {
        json.loads(line)["doc_id"]
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert docs_seen == {0, 1, 2}

    assert _FakeAdapter.instances and _FakeAdapter.instances[-1].calls >= 1
