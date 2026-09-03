from pathlib import Path

import pytest

from dataos.agent.runtime_compose import render, resolve
from dataos.agent.backend import system_prompt
from dataos.agent.config import AutoRunConfig, Budget
from dataos.agent.eval_freeze import freeze_eval
from dataos.agent.loop import _spec_from_node
from dataos_launch.task_config import load_task_contract

REPO = Path(__file__).resolve().parents[2]


def test_shipped_tasks_select_all_three_stage_drivers_explicitly():
    ptb = load_task_contract(REPO, "posttrainbench")
    assert {name: cfg["driver"] for name, cfg in ptb["stages"].items()} == {
        "data": "operator_toolbox",
        "train": "native",
        "eval": "posttrainbench",
    }
    assert ptb["stages"]["train"]["ownership"] == "agent"

    curation = load_task_contract(REPO, "curationbench")
    assert {name: cfg["driver"] for name, cfg in curation["stages"].items()} == {
        "data": "external_cli",
        "train": "external_cli",
        "eval": "external_cli",
    }
    assert curation["stages"]["train"]["ownership"] == "frozen"


def test_curation_contract_resolves_without_nonexistent_run_fields():
    resolved = resolve(
        {
            "environment": "example-slurm",
            "task": "curationbench",
            "base_model": "/models/smolvlm",
            "bench": "llava665k_smolvlm_8bench_10k_unlimited",
            "runs_base": "/runs",
            "deploy_root": "/deploy",
            "fixed_pool": "/datasets/llava665k",
            "eval_limit": -1,
            "resource": {
                "gpus": 1,
                "cpus": 8,
                "qos": "gpu_normal",
                "node": "gpu01",
                "partition": "gpu",
            },
        }
    )
    task = resolved["task"]
    assert task["agent"]["budget"] == {
        "unit": "iterations",
        "coordinator": "suite.py",
    }
    assert task["stages"]["train"]["driver"] == "external_cli"
    assert "dataos-train task-submit" in task["stages"]["train"]["command"]
    assert "--gpus 0" in task["stages"]["data"]["invoke"]
    assert "datacuration-bench eval" not in task["scoring"]["command"]
    assert "dataos-eval submit" in task["scoring"]["command"]
    assert task["scoring"]["format_contract"].endswith("/BENCHMARK.md")
    _, task_prompt = render(resolved)
    assert "/path/to/curation-bench/BENCHMARK.md" in task_prompt
    assert "/configs/tasks/curationbench//data2" not in task_prompt


def test_stage_contract_cannot_escape_task_directory(tmp_path):
    task_dir = tmp_path / "configs" / "tasks" / "bad"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "name: bad\nstages:\n"
        "  data: {stage: data, driver: x, ownership: agent}\n"
        "  train: {contract: ../../outside.yaml}\n"
        "  eval: {stage: eval, driver: x, ownership: frozen}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        load_task_contract(tmp_path, "bad")


def _curation_cfg() -> AutoRunConfig:
    cfg = AutoRunConfig(
        project_id="c",
        workspace="/tmp",
        target_bench="llava665k_smolvlm_8bench_10k_unlimited",
        h100_base_model="/models/root",
        eval_limit=-1,
        budget=Budget(max_rounds=11),
    )
    cfg.runtime_config = {"task": "curationbench", "cluster": {"gres": "gpu:1"}}
    return cfg


def test_curation_prompt_and_eval_driver_are_selected_by_task_config():
    cfg = _curation_cfg()
    leader = system_prompt("leader", cfg)
    experimenter = system_prompt("experimenter", cfg)
    assert "Curation-Bench run" in leader and "PostTrainBench" not in leader
    assert "11 official iterations" in leader and "10h budget" not in leader
    assert "datacuration-bench" in experimenter
    assert "LLaMA-Factory" in experimenter  # only in the explicit prohibition
    assert "budget is official iterations, never elapsed time" in experimenter
    assert "`script` is an object `{path, sha256}`" in experimenter
    assert freeze_eval(cfg=cfg) == {
        "provider": "external_cli",
        "task": "llava665k_smolvlm_8bench_10k_unlimited",
        "frozen": True,
    }


def test_curation_experiment_prompt_receives_the_claimed_official_target(tmp_path):
    from dataos.agent.roles import _task_prompt

    cfg = _curation_cfg()
    cfg.runtime_config.update(
        {
            "environment": "example-slurm",
            "kind": "slurm",
            "paths": {
                "deploy_root": "/deploy",
                "runs_base": "/runs",
                "fixed_pool": "/datasets/llava665k",
            },
            "cluster": {
                "gres": "gpu:1",
                "qos": "gpu_normal",
                "partition": "gpu",
                "nodelist": "gpu01",
                "cpus_per_task": 8,
            },
        }
    )
    spec = {
        "name": "random-baseline",
        "node_id": "n1",
        "parent_node_id": "n0",
        "init_model": "/models/root",
        "data": {},
        "train": {"model_name_or_path": "/models/root"},
        "subagent_brief": "baseline",
        "iteration": 1,
        "max_iterations": 11,
        "suite_dir": "/suite",
        "suite_target": "task_iter1",
        "suite_target_state": "active",
        "suite_run_dir": "/suite/task_iter1",
        "submission_path": "/suite/task_iter1/curated/final_submission",
    }

    text = _task_prompt(
        cfg=cfg,
        study_id="round-0000",
        spec=spec,
        exp_dir=tmp_path,
        eval_limit=-1,
    )

    assert "iteration 1/11" in text
    assert "Official suite: /suite" in text
    assert "never call `init` or `next`" in text
    assert "/suite/task_iter1/curated/final_submission" in text


def test_curation_tree_parent_is_strategy_only_and_model_resets_to_root():
    cfg = _curation_cfg()
    node = {
        "id": "n2",
        "parent": "n1",
        "experiment_name": "quality-filter",
        "init_model_path": "/models/parent-finetune",
        "plan": {"data": {"selector": "quality"}, "train": {}},
    }
    spec = _spec_from_node(node, {"provider": "external_cli"}, cfg)
    assert spec["init_model"] == "/models/root"
    assert spec["train"]["model_name_or_path"] == "/models/root"
    assert spec["data"] == {"selector": "quality"}
