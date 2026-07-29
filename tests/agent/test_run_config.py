import json

import pytest

from dataos.agent.run_config import PtbRunConfig, load_auto_run_config


def _run_json(run_dir, **over):
    data = {
        "run_id": "ptb-h100-qwen-gsm8k",
        "workspace": str(run_dir),
        "goal": "Improve GSM8K.",
        "bench": "gsm8k",
        "base_model_path": "/data/hcgu/hf_cache/Qwen/Qwen2.5-0.5B-Instruct",
        "time_budget_hours": 10,
        "resource": {"node": "gpu04", "gpus": 1, "cpus": 8},
        "environment": "PTB_H100_slurm",
        "task": "posttrainbench",
    }
    data.update(over)
    return data


def test_ptb_run_config_maps_to_auto_config(tmp_path):
    run_dir = tmp_path / "ptb-h100-qwen-gsm8k"
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(_run_json(run_dir)), encoding="utf-8")

    cfg = load_auto_run_config(config_path)

    assert cfg.project_id == "ptb-h100-qwen-gsm8k"
    assert cfg.workspace == str(tmp_path)
    assert cfg.target_bench == "gsm8k"
    assert cfg.h100_base_model == "/data/hcgu/hf_cache/Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.budget.max_wall_clock_s == 36000
    assert cfg.runtime_config["cluster"]["nodelist"] == "gpu04"
    assert cfg.runtime_config["cluster"]["gres"] == "gpu:1"
    assert cfg.runtime_config["cluster"]["cpus_per_task"] == 8
    assert (run_dir / "run_config.json").is_file()
    assert (run_dir / "resolved_config.json").is_file()


def test_paths_resolve_from_the_environment(tmp_path):
    """Run-level facts only; fixed_pool/train_env/eval_env come from configs/environments."""
    cfg = PtbRunConfig.from_dict(_run_json(tmp_path / "ptb-h100-qwen-gsm8k"))
    paths = cfg.runtime_config()["paths"]
    assert paths["fixed_pool"].endswith("posttrainbench_fixed_pool_v2")
    assert paths["train_env"].endswith("envs/sft/.venv")      # train.backends -> tools.sft_env
    assert paths["eval_env"].endswith("envs/eval/.venv")
    assert paths["runtime_root"].endswith("dataos_agentver/configs")
    assert cfg.runtime_config()["gateway"]["base_url"].startswith("https://")


def test_paths_block_is_rejected(tmp_path):
    """The old run-level `paths` block (and any unknown key) fails fast, not silently."""
    with pytest.raises(ValueError, match="unknown/forbidden keys: paths"):
        PtbRunConfig.from_dict(_run_json(tmp_path / "ptb-h100-qwen-gsm8k", paths={}))


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown/forbidden keys: typo_field"):
        PtbRunConfig.from_dict(_run_json(tmp_path / "ptb-h100-qwen-gsm8k", typo_field=1))


def test_environment_and_task_are_required(tmp_path):
    data = _run_json(tmp_path / "ptb-h100-qwen-gsm8k")
    del data["environment"]
    with pytest.raises(ValueError, match="environment"):
        PtbRunConfig.from_dict(data)
