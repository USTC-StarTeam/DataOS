from __future__ import annotations

import argparse
import importlib.util
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


def test_new_runs_keep_max_effort_and_separate_workspaces():
    from dataos.agent.run_config import PtbRunConfig
    from dataos_agent.providers.codex import CodexSpec
    from dataos_agent.types import AgentConfig

    workspaces = set()
    for name in ("qwen25vl3b", "llava7b"):
        path = REPO / f"configs/runs/curation-{name}-h100-max.json"
        config = PtbRunConfig.from_dict(yaml.safe_load(path.read_text()))
        assert config.agent.effort == "max"
        assert config.exec.resource.gpus == 2
        assert config.budget.iterations == 11
        workspaces.add(config.workspace)
    assert len(workspaces) == 2
    agent = AgentConfig(provider="codex", model="gpt-5.6-sol", effort="max")
    argv = CodexSpec().build_argv(agent, "test")
    assert "model_reasoning_effort=max" in argv
