from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, EvalArtifact, ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage


def _task(eval_config: dict) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "task-eval-api",
            "name": "eval-api",
            "goal": "run eval via api",
            "input_description": "hf dataset directory",
            "output_description": "artifacts",
            "constraints": {"dataset_format": "hf_datasets"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["dedup"],
            "train_config": {"mode": "stub"},
            "eval_config": eval_config,
        }
    )


def _dataset_artifact(sandbox: Sandbox) -> DatasetArtifact:
    artifact = DatasetArtifact(
        artifact_id="dataset-api-1",
        name="dataset",
        path=sandbox.artifact_dir("dataset", "dataset-api-1"),
        producer="test",
    )
    artifact.save(
        DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "instruction": ["What is 2+3?", "What is 5+5?"],
                        "response": ["#### 5", "#### 10"],
                        "answer": ["#### 5", "#### 10"],
                    }
                )
            }
        )
    )
    return artifact


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "model-api-1")
    model_dir.mkdir(parents=True, exist_ok=True)
    return ModelArtifact(
        artifact_id="model-api-1",
        name="model",
        path=model_dir,
        producer="test",
        meta={
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "api_base_url": "http://localhost:8000/v1",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        },
    )


def _eval_artifact(artifacts):
    for artifact in artifacts:
        if isinstance(artifact, EvalArtifact):
            return artifact
    raise AssertionError("EvalArtifact not found")


def _make_chat_response(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_openai_compatible_adapter_scores_gsm8k(tmp_path, monkeypatch):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(sandbox)
    model_artifact = _model_artifact(sandbox)
    task = _task(
        {
            "provider": "native",
            "task": {"name": "gsm8k"},
            "data": {"split": "test"},
            "adapter": {
                "name": "openai_compatible",
                "config": {
                    "base_url": "http://localhost:8000/v1",
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "api_key": "test-key",
                },
            },
            "request": {
                "type": "generate_until",
                "batch_size": 2,
                "max_new_tokens": 32,
            },
        }
    )

    api_calls: list[dict] = []
    reply_texts = ["The answer is #### 5", "I think it's #### 7"]

    class FakeResponse:
        status_code = 200

        def __init__(self, text: str):
            self._data = _make_chat_response(text)

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class FakeClient:
        def __init__(self, **kwargs):
            self.call_index = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json=None, headers=None):
            api_calls.append({"url": url, "json": json, "headers": headers})
            text = reply_texts[self.call_index]
            self.call_index += 1
            return FakeResponse(text)

    monkeypatch.setattr("dataos.eval.adapters.openai_compatible.httpx.Client", FakeClient)

    artifacts = run_eval_stage([dataset_artifact, model_artifact], sandbox=sandbox, task=task)
    eval_artifact = _eval_artifact(artifacts)

    assert eval_artifact.meta["provider"] == "native"
    assert eval_artifact.meta["task"] == "gsm8k"
    assert eval_artifact.meta["adapter"] == "openai_compatible"
    assert eval_artifact.meta["score"] == 0.5
    assert eval_artifact.meta["correct"] == 1
    assert eval_artifact.meta["num_examples"] == 2

    assert len(api_calls) == 2
    assert api_calls[0]["url"] == "http://localhost:8000/v1/chat/completions"
    assert api_calls[0]["json"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert api_calls[0]["headers"]["Authorization"] == "Bearer test-key"
