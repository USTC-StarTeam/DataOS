from __future__ import annotations

import argparse
import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]


def _runner():
    path = REPO / "configs/tasks/curationbench/eval/run_vlmeval.py"
    spec = importlib.util.spec_from_file_location("curation_eval_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("key", ["smolvlm", "qwen2.5-vl-3b-instruct", "llava-1.5-7b-hf"])
def test_model_resolution_uses_official_registry(tmp_path, monkeypatch, key):
    runner = _runner()
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump({"task": {"model_key": key}}))

    def resolve(command, **kwargs):
        assert command[0] == str(tmp_path / ".venv/bin/python")
        assert "benchmark.core.model_registry import get_model" in command[2]
        assert command[-1] == key
        assert kwargs["cwd"] == tmp_path
        return "official-adapter\n"

    monkeypatch.setattr(runner.subprocess, "check_output", resolve)
    assert runner._model_name(tmp_path, task) == "official-adapter"


@pytest.mark.parametrize("override", [None, "SmolVLM", "wrong-adapter"])
def test_inference_preserves_smolvlm_and_rejects_wrong_override(tmp_path, monkeypatch, override):
    runner = _runner()
    target = {"key": "iter1", "task_path": str(tmp_path / "task.yaml")}
    monkeypatch.setattr(runner, "_target", lambda *_: (target, tmp_path, ["MMStar"], tmp_path))
    monkeypatch.setattr(runner, "_model_name", lambda *_: "SmolVLM")
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    args = argparse.Namespace(
        suite_dir=tmp_path, model_path=tmp_path, harness_root=tmp_path,
        vlmeval_model=override, mode="infer",
    )
    if override == "wrong-adapter":
        with pytest.raises(ValueError, match="official model"):
            runner._run(args)
        assert not commands
    else:
        assert runner._run(args)["status"] == "completed"
        assert commands[0][commands[0].index("--model") + 1] == "SmolVLM"
        assert "--use-vllm" in commands[0]
        assert commands[1][1].endswith("validate_predictions.py")


def test_zero_exit_without_predictions_does_not_complete_target(tmp_path, monkeypatch):
    import subprocess

    runner = _runner()
    target = {"key": "iter1", "task_path": str(tmp_path / "task.yaml")}
    monkeypatch.setattr(runner, "_target", lambda *_: (target, tmp_path, ["MMStar"], tmp_path))
    monkeypatch.setattr(runner, "_model_name", lambda *_: "Qwen2.5-VL-3B-Instruct")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost")
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[1].endswith("validate_predictions.py"):
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(runner.subprocess, "run", run)
    args = argparse.Namespace(
        suite_dir=tmp_path, model_path=tmp_path, harness_root=tmp_path,
        vlmeval_model=None, mode="eval", judge_model="judge", workers=1,
        request_retries=1, timeout_seconds=120,
    )
    with pytest.raises(subprocess.CalledProcessError):
        runner._run(args)
    assert len(commands) == 1  # no judge work or SuiteController.mark_target_completed call


def test_qwen_inference_uses_nccl_entrypoint(tmp_path, monkeypatch):
    runner = _runner()
    target = {"key": "iter1", "task_path": str(tmp_path / "task.yaml")}
    monkeypatch.setattr(runner, "_target", lambda *_: (target, tmp_path, ["MMStar"], tmp_path))
    monkeypatch.setattr(runner, "_model_name", lambda *_: "Qwen2.5-VL-3B-Instruct")
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    args = argparse.Namespace(
        suite_dir=tmp_path, model_path=tmp_path, harness_root=tmp_path,
        vlmeval_model=None, mode="infer",
    )
    runner._run(args)
    assert commands[0][1].endswith("vlmeval_entry.py")
    assert commands[0][2].endswith("VLMEvalKit/run.py")


def test_codex_preserves_max_effort():
    from dataos_agent.providers.codex import CodexSpec
    from dataos_agent.types import AgentConfig

    agent = AgentConfig(provider="codex", model="gpt-5.6-sol", effort="max")
    argv = CodexSpec().build_argv(agent, "test")
    assert "model_reasoning_effort=max" in argv


@pytest.mark.parametrize("nested", [False, True])
def test_prediction_validation_requires_full_coverage(tmp_path, monkeypatch, nested):
    import pandas as pd

    validate = runpy.run_path(str(
        REPO / "configs/tasks/curationbench/eval/validate_predictions.py"
    ))["validate"]
    data = pd.DataFrame({"index": [0, 1]})
    module = types.ModuleType("vlmeval.dataset")
    module.build_dataset = lambda name: types.SimpleNamespace(data=data)
    monkeypatch.setitem(sys.modules, "vlmeval.dataset", module)
    directory = tmp_path / "model"
    if nested:
        directory /= "T20260908_Gtest"
    directory.mkdir(parents=True)
    (directory / "model_MMStar.xlsx").touch()
    valid = pd.DataFrame({"index": [0, 1], "prediction": ["A", "N/A"]})
    monkeypatch.setattr(pd, "read_excel", lambda *a, **kw: valid)
    validate(tmp_path, "model", ["MMStar"])
    for bad in (
        valid.iloc[:1],
        pd.DataFrame({"index": [0, 0], "prediction": ["A", "B"]}),
        pd.DataFrame({"index": [0, 1], "prediction": ["A", ""]}),
        pd.DataFrame({"index": [0, 1], "prediction": ["A", None]}),
        pd.DataFrame({"index": [0, 1], "prediction": ["A", "Failed to obtain answer via API."]}),
    ):
        monkeypatch.setattr(pd, "read_excel", lambda *a, **kw: bad)
        with pytest.raises(ValueError):
            validate(tmp_path, "model", ["MMStar"])
    with pytest.raises(ValueError, match="missing benchmark"):
        validate(tmp_path, "model", ["MMVet"])


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -1, 1001])
def test_invalid_native_scores_are_rejected(tmp_path, score):
    runner = _runner()
    (tmp_path / "results.json").write_text(json.dumps({"model": {"OCRBench": {
        "Final Score": score,
    }}}))
    with pytest.raises(ValueError):
        runner._validate_scores(tmp_path, "model", ["OCRBench"])


def test_missing_native_benchmark_is_rejected(tmp_path):
    runner = _runner()
    (tmp_path / "results.json").write_text(json.dumps({"model": {}}))
    with pytest.raises(ValueError):
        runner._validate_scores(tmp_path, "model", ["OCRBench"])


def test_nccl_workaround_preserves_other_llm_arguments(monkeypatch):
    calls = []
    module = types.ModuleType("vllm")
    module.LLM = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "vllm", module)
    runpy.run_path(str(REPO / "configs/tasks/curationbench/eval/vlmeval_entry.py"))[
        "configure_nccl"
    ]()
    module.LLM(model="qwen", tensor_parallel_size=2)
    assert calls == [{"model": "qwen", "tensor_parallel_size": 2, "disable_custom_all_reduce": True}]


def test_qwen_initialization_failure_exits_instead_of_retrying_each_benchmark(monkeypatch):
    module = types.ModuleType("vllm")

    def fail(**kwargs):
        raise RuntimeError("engine core initialization failed")

    module.LLM = fail
    monkeypatch.setitem(sys.modules, "vllm", module)
    runpy.run_path(str(REPO / "configs/tasks/curationbench/eval/vlmeval_entry.py"))[
        "configure_nccl"
    ]()
    with pytest.raises(SystemExit, match="initialization failed"):
        module.LLM(model="qwen")
