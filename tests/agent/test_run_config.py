import json

import pytest

from dataos.agent.run_config import PtbRunConfig, load_auto_run_config

V2 = "/data2/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v2"


def _run_json(ws, **over):
    data = {
        "goal": "Improve GSM8K.",
        "workspace": str(ws),
        "environment": "PTB_H100_slurm",
        "task": "posttrainbench",
        "bench": "gsm8k",
        "inputs": {
            "base_model_path": "/data/hcgu/hf_cache/Qwen/Qwen2.5-0.5B-Instruct",
            "fixed_pool": V2,
        },
        "exec": {
            "deploy_root": "/data/hcgu/users/tjh/dataos_agentver",
            "runs_base": "/data/hcgu/users/tjh/dataos_agentver_runs",
            "resource": {"node": "gpu04", "gpus": 1, "cpus": 8},
        },
        "budget": {"time_hours": 10, "eval_samples": -1},
    }
    data.update(over)
    return data


def test_ptb_run_config_maps_to_auto_config(tmp_path):
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(_run_json(tmp_path / "myrun")), encoding="utf-8")

    cfg = load_auto_run_config(config_path)

    assert cfg.project_id == "myrun"  # identity = workspace basename
    assert cfg.workspace == str(tmp_path)
    assert cfg.target_bench == "gsm8k"
    assert cfg.h100_base_model == "/data/hcgu/hf_cache/Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.budget.max_wall_clock_s == 36000
    assert cfg.runtime_config["cluster"]["nodelist"] == "gpu04"
    run_dir = tmp_path / "myrun"
    assert (run_dir / "run_config.json").is_file()
    assert (run_dir / "resolved_config.json").is_file()


def test_paths_resolve_with_derivations(tmp_path):
    cfg = PtbRunConfig.from_dict(_run_json(tmp_path / "myrun"))
    paths = cfg.runtime_config()["paths"]
    assert paths["fixed_pool"] == V2
    assert paths["deploy_root"].endswith("dataos_agentver")
    assert paths["runs_base"].endswith("dataos_agentver_runs")
    # derived, not declared
    assert paths["runtime_root"].endswith("dataos_agentver/configs")
    assert paths["benchmarks"].endswith("configs/tasks/posttrainbench/benchmarks")
    assert paths["train_env"].endswith("envs/sft/.venv")
    assert paths["train_cli_env"].endswith("envs/train-cli/.venv")
    gateway = cfg.runtime_config()["gateway"]
    assert gateway["base_url"].startswith("https://") and "model" not in gateway


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown/forbidden keys"):
        PtbRunConfig.from_dict(_run_json(tmp_path / "myrun", paths={}))
    with pytest.raises(ValueError, match="unknown/forbidden keys"):
        PtbRunConfig.from_dict(_run_json(tmp_path / "myrun", run_id="x"))


def test_required_fields(tmp_path):
    for missing in ("workspace", "environment", "task", "inputs", "exec", "budget"):
        data = _run_json(tmp_path / "myrun")
        del data[missing]
        with pytest.raises(ValueError, match=missing):
            PtbRunConfig.from_dict(data)
    data = _run_json(tmp_path / "myrun")
    del data["inputs"]["fixed_pool"]
    with pytest.raises(ValueError, match="fixed_pool"):
        PtbRunConfig.from_dict(data)


def test_fresh_instance_suffixes_taken_workspace(tmp_path):
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(_run_json(tmp_path / "myrun")), encoding="utf-8")
    (tmp_path / "myrun").mkdir()
    (tmp_path / "myrun(1)").mkdir()

    cfg = load_auto_run_config(config_path)

    assert cfg.project_id == "myrun(2)"
    assert (tmp_path / "myrun(2)" / "run_config.json").is_file()


def test_resume_requires_existing_workspace(tmp_path):
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(_run_json(tmp_path / "myrun")), encoding="utf-8")
    with pytest.raises(ValueError, match="--resume"):
        load_auto_run_config(config_path, resume=True)

    (tmp_path / "myrun").mkdir()
    cfg = load_auto_run_config(config_path, resume=True)
    assert cfg.project_id == "myrun"  # same instance, no suffix


def test_agent_block_configures_provider(tmp_path, monkeypatch):
    """The optional `agent` block sets provider/model; precedence: JSON > env > default."""
    monkeypatch.delenv("DATAOS_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("DATAOS_AGENT_MODEL", raising=False)
    base = _run_json(tmp_path / "myrun")

    cfg = PtbRunConfig.from_dict(base).to_auto_config(resolved_config_path=tmp_path / "r.json")
    assert cfg.agent_provider == "kimi" and cfg.agent_model is None  # defaults

    cfg = PtbRunConfig.from_dict(
        _run_json(tmp_path / "myrun", agent={"provider": "claude"})
    ).to_auto_config(resolved_config_path=tmp_path / "r.json")
    assert cfg.agent_provider == "claude" and cfg.agent_model is None  # CLI default model

    cfg = PtbRunConfig.from_dict(
        _run_json(tmp_path / "myrun", agent={"provider": "claude", "model": "claude-opus-4-8"})
    ).to_auto_config(resolved_config_path=tmp_path / "r.json")
    assert cfg.agent_model == "claude-opus-4-8"

    # env var beats nothing but the JSON default; JSON beats env
    monkeypatch.setenv("DATAOS_AGENT_PROVIDER", "qwen")
    cfg = PtbRunConfig.from_dict(base).to_auto_config(resolved_config_path=tmp_path / "r.json")
    assert cfg.agent_provider == "qwen"
    cfg = PtbRunConfig.from_dict(
        _run_json(tmp_path / "myrun", agent={"provider": "claude"})
    ).to_auto_config(resolved_config_path=tmp_path / "r.json")
    assert cfg.agent_provider == "claude"

    with pytest.raises(ValueError, match="unknown keys"):
        PtbRunConfig.from_dict(_run_json(tmp_path / "myrun", agent={"provder": "claude"}))
