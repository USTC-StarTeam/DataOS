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
            "resource": {"gpus": 1, "cpus": 8},
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
    assert cfg.runtime_config["cluster"]["nodelist"] is None  # unpinned: Slurm picks
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
    assert paths["train_env"].endswith("envs/dataos_sft/.venv")  # sft 基线的 DataOS 派生 env
    # train-cli 已取消——CLI 由 dataos_sft 承载(回退链:第一个 provides dataos-train 的 env)
    assert paths["train_cli_env"].endswith("envs/dataos_sft/.venv")
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


def test_agent_idle_timeout_s(tmp_path):
    """agent.idle_timeout_s: default 1800, overridable, 0 disables (None downstream)."""
    cfg = PtbRunConfig.from_dict(
        _run_json(tmp_path / "myrun")
    ).to_auto_config(resolved_config_path=tmp_path / "r.json")
    assert cfg.agent_idle_timeout_s == 1800.0  # default

    cfg = PtbRunConfig.from_dict(
        _run_json(tmp_path / "myrun", agent={"idle_timeout_s": 600})
    ).to_auto_config(resolved_config_path=tmp_path / "r.json")
    assert cfg.agent_idle_timeout_s == 600.0

    cfg = PtbRunConfig.from_dict(
        _run_json(tmp_path / "myrun", agent={"idle_timeout_s": 0})
    ).to_auto_config(resolved_config_path=tmp_path / "r.json")
    assert cfg.agent_idle_timeout_s == 0.0  # build_agent maps 0 -> None (disabled)

    with pytest.raises(ValueError, match="idle_timeout_s"):
        PtbRunConfig.from_dict(_run_json(tmp_path / "myrun", agent={"idle_timeout_s": -5}))


def test_node_is_optional():
    """resource.node can be omitted — Slurm then picks any free node."""
    data = _run_json("/tmp/x")
    assert "node" not in data["exec"]["resource"]
    cfg = PtbRunConfig.from_dict(data)
    assert cfg.exec.resource.node is None
    assert cfg.runtime_config()["cluster"]["nodelist"] is None
    # explicit pin still works
    data2 = _run_json("/tmp/x")
    data2["exec"]["resource"]["node"] = "gpu01"
    assert PtbRunConfig.from_dict(data2).exec.resource.node == "gpu01"


def test_budget_ideal_time_excludes_queue_credit():
    from dataos.agent.budget import stop_reason
    from dataos.agent.config import Budget

    b = Budget(max_wall_clock_s=100.0)
    # 150s elapsed but 80s was queue wait -> ideal 70s < 100 -> no stop
    assert stop_reason(started_at=0, budget=b, now=150.0, queue_credit_s=80.0) is None
    # 150s elapsed with 40s queue -> ideal 110s >= 100 -> stop
    assert stop_reason(started_at=0, budget=b, now=150.0, queue_credit_s=40.0) == "budget:wall_clock"


def test_eval_samples_auto_mode(tmp_path):
    """budget.eval_samples: -2 = auto (experimenter picks the eval size per experiment)."""
    cfg = PtbRunConfig.from_dict(
        _run_json(tmp_path / "myrun", budget={"time_hours": 1, "eval_samples": -2})
    ).to_auto_config(resolved_config_path=tmp_path / "r.json")
    assert cfg.eval_limit == -2

    with pytest.raises(ValueError, match="eval_samples"):
        PtbRunConfig.from_dict(
            _run_json(tmp_path / "myrun", budget={"time_hours": 1, "eval_samples": -3})
        )


def test_auto_eval_limit_prompts(tmp_path):
    """-2 (auto): the LEADER picks each experiment's eval size via plan.eval_limit; the
    experimenter only executes the number; the task constraint says the plan owns it."""
    from dataos import prompt as _prompt
    from dataos.agent.runtime_compose import resolve, render

    text = _prompt.render(
        "task/experimenter", exp_name="e", bench="gsm8k", profile="formal",
        parent_model="/m", root_base_model="/m", node_id="n1", parent_node_id="n0",
        eval_limit=256, is_smoke=False, recipe_json="{}", train_json="{}",
        train_spec_json="", subagent_brief="b", runtime_config_json="{}",
        run_root="/r", report_path="/r/report.json", state_path="/r/s.json",
        env_ops="", task_def="", stage_blocks="",
    )
    assert "eval limit: 256" in text and "set by the plan" in text

    rc = {"environment": "PTB_H100_slurm", "task": "posttrainbench", "base_model": "/m",
          "bench": "gsm8k", "runs_base": "/r", "deploy_root": "/deploy",
          "resource": {"gpus": 1, "cpus": 8, "node": None},
          "eval_limit": -2, "fixed_pool": "/p"}
    env_ops, task_def = render(resolve(rc))
    assert "评多少由 plan 决定" in task_def
    rc["eval_limit"] = -1
    _, task_def_full = render(resolve(rc))
    assert "--limit 已固定为 -1" in task_def_full
