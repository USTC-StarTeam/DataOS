from __future__ import annotations

import json

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, EvalArtifact, ModelArtifact, ResultArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.eval.tasks.climaqa import _parse_options, _parse_prediction
from dataos.eval.types import ModelResponse


OPTIONS = (
    "a) Due to evaporation below cloud. -------------------- "
    "b) As snowflakes become more compact. -------------------- "
    "c) Due to melting snowflakes acquiring a liquid layer. -------------------- "
    "d) Because ice crystal concentration increases."
)


def _task(eval_config: dict) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "task-climaqa-eval",
            "name": "climaqa-eval",
            "goal": "run climaqa eval",
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
        artifact_id="dataset-climaqa-1",
        name="dataset",
        path=sandbox.artifact_dir("dataset", "dataset-climaqa-1"),
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "model-climaqa-1")
    model_dir.mkdir(parents=True, exist_ok=True)
    return ModelArtifact(
        artifact_id="model-climaqa-1",
        name="model",
        path=model_dir,
        producer="test",
        meta={"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "mode": "base_model_eval"},
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


def test_eval_model_climaqa_scores_predictions_and_persists_metadata(tmp_path, monkeypatch):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "Question": [
                            "Why does radar reflectivity increase?",
                            None,
                            "What is responsible for tropical cyclone circulation?",
                        ],
                        "question": [
                            None,
                            "How did anthropogenic aerosol impact the monsoon?",
                            None,
                        ],
                        "Options": [OPTIONS, None, OPTIONS],
                        "options": [
                            None,
                            "A. It increased precipitation\nB. It strengthened circulation\n"
                            "C. It enhanced monsoon circulation\nD. It weakened the monsoon",
                            None,
                        ],
                        "Answer": ["C", None, "D"],
                        "answer": [None, "d", None],
                        "Complexity": ["REASONING", None, "BASE"],
                        "complexity": [None, "HYPOTHETICAL", None],
                    }
                )
            }
        ),
    )
    model_artifact = _model_artifact(sandbox)
    task = _task(
        {
            "provider": "native",
            "task": {"name": "climaqa", "config": {}},
            "data": {"split": "test"},
            "adapter": {"name": "transformers", "config": {}},
            "request": {
                "type": "generate_until",
                "batch_size": 2,
                "max_new_tokens": 4,
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
        texts = ["Answer: C", "选 D", "I choose A"]
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
    request_artifact = _result_artifact(artifacts, "eval_requests")
    response_artifact = _result_artifact(artifacts, "eval_responses")
    sample_artifact = _result_artifact(artifacts, "eval_samples")

    assert eval_artifact.meta["provider"] == "native"
    assert eval_artifact.meta["task"] == "climaqa"
    assert eval_artifact.meta["adapter"] == "transformers"
    assert eval_artifact.meta["request_type"] == "generate_until"
    assert eval_artifact.meta["metric"] == "accuracy"
    assert eval_artifact.meta["dataset_split"] == "test"
    assert eval_artifact.meta["score"] == 2 / 3
    assert eval_artifact.meta["accuracy"] == 2 / 3
    assert eval_artifact.meta["exact_match"] == 2 / 3
    assert eval_artifact.meta["correct"] == 2
    assert eval_artifact.meta["num_examples"] == 3
    assert eval_artifact.meta["invalid_predictions"] == 0
    assert eval_artifact.meta["trace_artifact_ids"] == {
        "requests": request_artifact.artifact_id,
        "responses": response_artifact.artifact_id,
        "samples": sample_artifact.artifact_id,
    }

    summary = json.loads((eval_artifact.path / "results.json").read_text(encoding="utf-8"))
    assert summary["task_version"] == "v1"
    assert summary["metric"] == "accuracy"

    request_lines = (request_artifact.path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    request_record = json.loads(request_lines[0])
    assert request_record["choices"] == ["a", "b", "c", "d"]
    assert "Why does radar reflectivity increase?" in request_record["prompt"]
    assert "Only output one letter: a, b, c, or d." in request_record["prompt"]

    sample_lines = (sample_artifact.path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    first_sample = json.loads(sample_lines[0])
    assert first_sample["predicted_answer"] == "c"
    assert first_sample["reference_answer"] == "c"
    assert first_sample["is_correct"] is True
    assert first_sample["meta"]["complexity"] == "REASONING"
    assert first_sample["meta"]["raw_options"] == OPTIONS
    assert first_sample["meta"]["parsed_options"] == {
        "a": "Due to evaporation below cloud.",
        "b": "As snowflakes become more compact.",
        "c": "Due to melting snowflakes acquiring a liquid layer.",
        "d": "Because ice crystal concentration increases.",
    }


def test_climaqa_prediction_parser_handles_common_answer_forms():
    assert _parse_prediction("c") == "c"
    assert _parse_prediction("Answer: C") == "c"
    assert _parse_prediction("答案是 C") == "c"
    assert _parse_prediction("选 C") == "c"
    assert _parse_prediction("The correct option is d.") == "d"
    assert _parse_prediction("A or B") is None


def test_climaqa_option_parser_ignores_citation_initials_inside_options():
    options = (
        "a) Riahi, K., van Vuuren, D.P., O'Neill, B.C.\n"
        "--------------------\n"
        "b) Rafaj, P., Amann, M., 2014. Changes in European emissions.\n"
        "--------------------\n"
        "c) Reis, S., Grennfelt, P., 2012. From acid rain to climate change.\n"
        "--------------------\n"
        "d) Rafaj, P., Amann, M., 2018. Decomposing air pollutant emissions."
    )

    parsed = _parse_options(options, 0)

    assert parsed["a"].startswith("Riahi, K.")
    assert parsed["b"].startswith("Rafaj, P., Amann, M., 2014")
    assert parsed["c"].startswith("Reis, S.")
    assert parsed["d"].startswith("Rafaj, P., Amann, M., 2018")
