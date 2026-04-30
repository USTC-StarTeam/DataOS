from __future__ import annotations

import json

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, EvalArtifact, ModelArtifact, ResultArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.eval.registry import get_eval_task
from dataos.eval.tasks.code_execution import run_python_code_tests
from dataos.eval.tasks.livecodebench import LiveCodeBenchTask
from dataos.eval.types import ModelResponse


def _task(eval_config: dict) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "task-livecodebench-eval",
            "name": "livecodebench-eval",
            "goal": "run livecodebench eval",
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
        artifact_id="dataset-livecodebench-1",
        name="dataset",
        path=sandbox.artifact_dir("dataset", "dataset-livecodebench-1"),
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "model-livecodebench-1")
    model_dir.mkdir(parents=True, exist_ok=True)
    return ModelArtifact(
        artifact_id="model-livecodebench-1",
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


def test_livecodebench_executes_unit_tests_inside_sandbox_tmp(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    result = run_python_code_tests(
        code="""\
def add(a, b):
    return a + b
""",
        tests=["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
        sandbox_tmp_dir=sandbox.tmp_dir,
        sample_id="sample-1",
        timeout_seconds=2,
    )

    assert result.passed is True
    assert result.passed_tests == 2
    assert result.failed_tests == 0
    assert result.execution_error is None
    assert (sandbox.tmp_dir / "livecodebench").exists()


def test_livecodebench_blocks_network_and_writes_outside_sandbox_tmp(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    outside_path = tmp_path / "outside.txt"
    result = run_python_code_tests(
        code=f"""\
OUTSIDE_PATH = {str(outside_path)!r}


def write_outside():
    with open(OUTSIDE_PATH, "w", encoding="utf-8") as handle:
        handle.write("bad")


def connect_network():
    import socket

    socket.create_connection(("example.com", 80), timeout=1)
""",
        tests=["write_outside()", "connect_network()"],
        sandbox_tmp_dir=sandbox.tmp_dir,
        sample_id="restricted",
        timeout_seconds=2,
    )

    assert result.passed is False
    assert result.passed_tests == 0
    assert result.failed_tests == 2
    assert result.execution_error is not None
    assert "writes outside the sandbox tmp directory are disabled" in result.execution_error
    assert "network and subprocess access are disabled" in result.execution_error
    assert not outside_path.exists()


def test_livecodebench_executes_stdin_stdout_tests(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    result = run_python_code_tests(
        code="""\
import sys

value = int(sys.stdin.read().strip())
print(value + 1)
""",
        tests=[{"input": "41\n", "output": "42\n"}],
        sandbox_tmp_dir=sandbox.tmp_dir,
        sample_id="stdio",
        timeout_seconds=2,
    )

    assert result.passed is True
    assert result.passed_tests == 1
    assert result.failed_tests == 0
    assert result.execution_error is None


def test_eval_model_livecodebench_scores_code_and_persists_metadata(tmp_path, monkeypatch):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "question": [
                            "Implement add(a, b).",
                            None,
                        ],
                        "description": [
                            None,
                            "Implement double(x).",
                        ],
                        "starter_code": [
                            "def add(a, b):\n    pass",
                            "def double(x):\n    pass",
                        ],
                        "public_tests": [
                            [
                                "assert add(1, 2) == 3",
                                "assert add(-1, 1) == 0",
                            ],
                            [
                                "assert double(2) == 4",
                            ],
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
            "task": {"name": "livecodebench", "config": {"timeout_seconds": 2}},
            "data": {"split": "test"},
            "adapter": {"name": "transformers", "config": {}},
            "request": {
                "type": "generate_until",
                "batch_size": 2,
                "max_new_tokens": 128,
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
        texts = [
            "```python\ndef add(a, b):\n    return a + b\n```",
            "def double(x)\n    return x * 2",
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
    request_artifact = _result_artifact(artifacts, "eval_requests")
    response_artifact = _result_artifact(artifacts, "eval_responses")
    sample_artifact = _result_artifact(artifacts, "eval_samples")

    assert eval_artifact.meta["provider"] == "native"
    assert eval_artifact.meta["task"] == "livecodebench"
    assert eval_artifact.meta["adapter"] == "transformers"
    assert eval_artifact.meta["request_type"] == "generate_until"
    assert eval_artifact.meta["metric"] == "pass@1"
    assert eval_artifact.meta["dataset_split"] == "test"
    assert eval_artifact.meta["score"] == 0.5
    assert eval_artifact.meta["pass@1"] == 0.5
    assert eval_artifact.meta["passed_tests"] == 2
    assert eval_artifact.meta["failed_tests"] == 1
    assert eval_artifact.meta["execution_error"] == 1
    assert eval_artifact.meta["trace_artifact_ids"] == {
        "requests": request_artifact.artifact_id,
        "responses": response_artifact.artifact_id,
        "samples": sample_artifact.artifact_id,
    }

    summary = json.loads((eval_artifact.path / "results.json").read_text(encoding="utf-8"))
    assert summary["task_version"] == "v1"
    assert summary["metric"] == "pass@1"

    request_lines = (request_artifact.path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    first_request = json.loads(request_lines[0])
    assert "Implement add(a, b)." in first_request["prompt"]
    assert "Starter code:" in first_request["prompt"]
    assert first_request["meta"]["tests_field"] == "public_tests"
    assert first_request["meta"]["test_count"] == 2

    sample_lines = (sample_artifact.path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(sample_lines) == 2
    first_sample = json.loads(sample_lines[0])
    second_sample = json.loads(sample_lines[1])
    assert first_sample["is_correct"] is True
    assert first_sample["metrics"]["pass@1"] == 1.0
    assert second_sample["is_correct"] is False
    assert second_sample["metrics"]["execution_error"] == 1.0
    assert "SyntaxError" in second_sample["meta"]["execution_error"]

    response_lines = (response_artifact.path / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(response_lines) == 2
    assert (sandbox.tmp_dir / "livecodebench").exists()


def test_livecodebench_is_registered():
    assert isinstance(get_eval_task("livecodebench"), LiveCodeBenchTask)
