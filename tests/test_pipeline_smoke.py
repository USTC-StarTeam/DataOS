from __future__ import annotations

import os

import pytest
from datasets import Dataset

from dataos.core.artifact import DatasetArtifact, EvalArtifact, ExperimentArtifact, ModelArtifact
from dataos.eval.types import AdapterCallContext, EvalRequest, ModelResponse
from dataos.runtime.runner import run_task

pytest.importorskip("lm_eval")


class FakeLLMClient:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
        del system_prompt, user_prompt, temperature
        return self.responses.pop(0)


class _FakeAdapter:
    name = "transformers"

    def generate_until(
        self,
        model_artifact,
        requests: list[EvalRequest],
        config: AdapterCallContext,
    ) -> list[ModelResponse]:
        del model_artifact, config
        return [
            ModelResponse(
                sample_id=request.sample_id,
                index=request.index,
                request_type="generate_until",
                text="Step-by-step reasoning gives us 18. #### 18",
            )
            for request in requests
        ]


def test_run_task_smoke(tmp_path, monkeypatch):
    if not os.environ.get("HF_ENDPOINT") and not os.environ.get("HF_DATASETS_OFFLINE"):
        pytest.skip("HF_ENDPOINT not set; skip to avoid network reach to huggingface.co")

    dataset = Dataset.from_dict(
        {
            "instruction": ["hello", "", "world"],
            "response": ["hi", "", "earth"],
        }
    )
    source_dir = tmp_path / "source_dataset"
    dataset.save_to_disk(str(source_dir))

    task_template = {
        "task_id": "task-smoke",
        "name": "smoke",
        "goal": "run full loop",
        "input_description": "hf dataset directory",
        "output_description": "artifacts",
        "constraints": {"dataset_format": "hf_datasets"},
        "budget": {"max_experiments": 1},
        "max_rounds": 1,
        "pipeline_hint": ["dedup", "sample"],
        "train_config": {"mode": "stub"},
        "eval_config": {
            "tasks": ["gsm8k"],
            "limit": 2,
            "num_fewshot": 0,
            "batch_size": 2,
            "max_new_tokens": 16,
            "apply_chat_template": False,
            "adapter": "transformers",
        },
    }

    monkeypatch.setattr(
        "dataos.eval.runner._make_adapter",
        lambda name, config: _FakeAdapter(),
    )

    llm = FakeLLMClient(
        [
            {
                "steps": [
                    {"operator_name": "dedup", "params": {"keys": ["instruction"]}},
                    {"operator_name": "sample", "params": {"max_train": 10}},
                ],
                "reasoning_summary": "Run the standard pipeline.",
                "expected_gain": "Produce deduped, capped data before the fixed runtime tools run.",
            },
            {
                "summary": "The run completed.",
                "issues": [],
                "next_hypothesis": "Try a different sample cap next round.",
                "recommended_changes": ["Adjust sample max_train if quality remains stable."],
            },
        ]
    )

    result = run_task(
        task_template=task_template,
        dataset_path=source_dir,
        workspace_dir=tmp_path / "workspace",
        llm_client=llm,
    )

    assert (result.experiment.sandbox.root / "report.json").exists()
    assert (result.experiment.sandbox.root / "review.json").exists()
    assert any(
        isinstance(artifact, DatasetArtifact)
        and artifact.artifact_id != result.initial_artifact.artifact_id
        for artifact in result.experiment.artifacts
    )
    assert any(isinstance(artifact, ModelArtifact) for artifact in result.experiment.artifacts)
    eval_artifacts = [
        artifact for artifact in result.experiment.artifacts if isinstance(artifact, EvalArtifact)
    ]
    assert eval_artifacts, "expected at least one EvalArtifact from run_eval_stage"
    eval_artifact = eval_artifacts[-1]
    assert eval_artifact.meta["backend"] == "lm_eval"
    assert eval_artifact.meta["tasks"] == ["gsm8k"]
    assert eval_artifact.meta["adapter"] == "transformers"
    assert isinstance(result.experiment_artifact, ExperimentArtifact)
    assert result.archived_experiment_artifact.exists()
