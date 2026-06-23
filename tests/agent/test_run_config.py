import json

from dataos.agent.run_config import load_auto_run_config


def test_ptb_run_config_maps_to_auto_config(tmp_path):
    run_dir = tmp_path / "ptb-h100-qwen-gsm8k"
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "ptb-h100-qwen-gsm8k",
                "workspace": str(run_dir),
                "goal": "Improve GSM8K.",
                "bench": "gsm8k",
                "base_model_path": "/data/hcgu/hf_cache/Qwen/Qwen2.5-0.5B-Instruct",
                "time_budget_hours": 10,
                "resource": {"node": "gpu04", "gpus": 1, "cpus": 8},
                "paths": {
                    "fixed_pool": "/data/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v1_full",
                    "conda_sh": "/data/software/miniconda3/etc/profile.d/conda.sh",
                    "train_env": "/data/hcgu/.conda/envs/train",
                    "eval_env": "/data/hcgu/.conda/envs/lm-eval",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_auto_run_config(config_path)

    assert cfg.project_id == "ptb-h100-qwen-gsm8k"
    assert cfg.workspace == str(tmp_path)
    assert cfg.target_bench == "gsm8k"
    assert cfg.train_executor == "subagent"
    assert cfg.h100_base_model == "/data/hcgu/hf_cache/Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.budget.max_wall_clock_s == 36000
    assert cfg.runtime_config["cluster"]["nodelist"] == "gpu04"
    assert cfg.runtime_config["cluster"]["gres"] == "gpu:1"
    assert cfg.runtime_config["cluster"]["cpus_per_task"] == 8
    assert (run_dir / "run_config.json").is_file()
    assert (run_dir / "resolved_config.json").is_file()
