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
        "data": "dataos-data",
        "train": "native",
        "eval": "posttrainbench",
    }
    assert ptb["stages"]["train"]["ownership"] == "agent"

    curation = load_task_contract(REPO, "curationbench")
    assert {name: cfg["driver"] for name, cfg in curation["stages"].items()} == {
        "data": "dataos-data",
        "train": "external_cli",
        "eval": "external_cli",
    }
    assert curation["stages"]["train"]["ownership"] == "frozen"
    assert curation["stages"]["data"]["toolbox"] == "dataos-operator"


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
    assert task["stages"]["data"]["toolbox"] == "dataos-operator"
    assert task["stages"]["data"]["dataset_backend"] == "indexed"
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
    assert "CURATION TOOLBOX" in text
    assert "weighted_reservoir_selector" in text
    assert "length_filter" not in text
    assert "convert to Alpaca" in text
    assert "check objective data feasibility and integrity" in text
    assert "There is no separate sample-quality admission gate" in text
    assert "supersedes sample-quality audit gates in older plans" in text
    assert "official contamination audit remain" in text
    assert "stable pool row indices or full-row hashes" in text
    assert "later confirmation-only plans are forbidden" in text
    assert "must reuse durable handles/results" in text
    assert "baseline/parent data is a comparison reference" in text
    assert "Do not introduce an unrequested retention quota" in text
    assert "reuse running/completed" in text
    assert "those rules govern the next newly planned selection" in text


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


def test_curation_leader_receives_the_operator_catalog():
    from dataos.agent.planner import _build_prompt

    text = _build_prompt(
        context={"round": 2},
        catalog=[
            {
                "name": "weighted_reservoir_selector",
                "kind": "selector",
                "params": [{"name": "sample_size"}],
            }
        ],
        catalog_path="/workspace/operator_catalog.json",
        template="curationbench/leader/turn/plan",
    )

    assert "weighted_reservoir_selector" in text
    assert "/workspace/operator_catalog.json" in text
    assert "Do not spend" in text
    assert "baseline replication, seed-only repetition, noise estimation" in text
    assert "primary intervention" in text
    assert "expected data change" in text
    assert "full frozen candidate pool" in text
    assert "Do not default to retaining a percentage" in text
    assert "do not force zero overlap" in text
    assert "Measure natural overlap after selection" in text
    assert "Anchor refresh remains available when evidence warrants" in text
    assert "Do not add a separate sample-quality acceptance stage" in text
    assert "Require bounded\n  sample checks" not in text


def test_curation_review_requires_item_evidence_without_confirmation_rounds():
    from dataos import prompt

    text = prompt.render("curationbench/experimenter/turn/stage_review")
    assert "per-item predictions" in text
    assert "incorrect examples" in text and "correct controls" in text
    assert "observations, causal hypotheses and unresolved uncertainty" in text
    assert "Do not recommend a new official" in text
    assert "noise estimation or confirmation" in text
    assert "out of curation code and" in text
    assert "training submissions" in text
    assert "canonical score is present and finite" in text
    assert "Evaluate the selection criterion" in text
    assert "not evidence that its individual rows should be retained" in text
    assert "arbitrary row-identity preservation" in text
    assert "not a pre-training sample-quality gate" in text


def test_capacity_repair_does_not_reintroduce_sample_quality_gates():
    from dataos import prompt

    text = prompt.render("curationbench/leader/turn/repair", context_json="{}", catalog_json="[]",
                         validation_errors_json=None)
    assert "objective capacity, exact row count and schema" in text
    assert "Do not introduce or preserve a separate sample-quality admission gate" in text
    assert "perform its bounded sample audit" not in text
