from __future__ import annotations

from datasets import Dataset

from dataos.core.artifact import DatasetArtifact, EvalArtifact, ExperimentArtifact, ModelArtifact
from dataos.runtime.runner import run_task


class FakeLLMClient:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
        del system_prompt, user_prompt, temperature
        return self.responses.pop(0)


def test_run_task_smoke(tmp_path):
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
        "pipeline_hint": ["clean_data", "synthesize_data"],
        "train_config": {"mode": "stub"},
        "eval_config": {"provider": "stub", "provider_config": {"score": 0.5}},
    }
    llm = FakeLLMClient(
        [
            {
                "steps": [
                    {"operator_name": "clean_data", "params": {}},
                    {"operator_name": "synthesize_data", "params": {"augment_ratio": 0.5}},
                ],
                "reasoning_summary": "Run the standard pipeline.",
                "expected_gain": "Produce cleaned data before the fixed runtime tools run.",
            },
            {
                "summary": "The run completed.",
                "issues": [],
                "next_hypothesis": "Try a larger augmentation ratio next round.",
                "recommended_changes": ["Increase synthesize_data augment_ratio if quality remains stable."],
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
    assert any(isinstance(artifact, DatasetArtifact) and artifact.artifact_id != result.initial_artifact.artifact_id for artifact in result.experiment.artifacts)
    assert any(isinstance(artifact, ModelArtifact) for artifact in result.experiment.artifacts)
    assert any(isinstance(artifact, EvalArtifact) for artifact in result.experiment.artifacts)
    assert isinstance(result.experiment_artifact, ExperimentArtifact)
    assert result.archived_experiment_artifact.exists()
