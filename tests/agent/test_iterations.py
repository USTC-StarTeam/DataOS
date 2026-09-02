import json

from dataos.agent import iterations, runtime
from dataos.agent.config import AutoRunConfig, Budget


def _cfg() -> AutoRunConfig:
    cfg = AutoRunConfig(
        project_id="curation-run",
        workspace="/workspace",
        target_bench="llava665k_smolvlm_8bench_10k_unlimited",
        h100_base_model="/models/root",
        eval_limit=-1,
        budget=Budget(max_rounds=11),
    )
    cfg.runtime_config = {
        "environment": "example-slurm",
        "task": "curationbench",
        "paths": {"deploy_root": "/deploy", "runs_base": "/runs"},
        "cluster": {"gres": "gpu:1"},
    }
    return cfg


def test_claim_maps_a_durable_round_to_one_official_target(monkeypatch):
    captured = []

    def fake_exec(cfg, argv, timeout=120):  # noqa: ARG001
        captured.append(argv)
        return json.dumps(
            {
                "status": "target_activated",
                "iteration": 3,
                "target_key": "llava665k_smolvlm_8bench_10k_unlimited_iter3",
                "target_state": "active",
                "suite_dir": "/runs/curation-run/_curation_suite/suite",
                "run_dir": "/runs/curation-run/_curation_suite/suite/task_iter3",
                "default_submission_path": "/runs/submission",
            }
        )

    monkeypatch.setattr(runtime, "exec_host_command", fake_exec)
    result = iterations.claim(_cfg(), {"round": 2})

    assert result["iteration"] == 3
    command = captured[0]
    assert command[-2:] == ["--expected-iteration", "3"]
    assert "--iterations" in command and command[command.index("--iterations") + 1] == "11"
    assert "/deploy/configs/tasks/curationbench/suite.py" in command


def test_non_iteration_task_has_no_suite_coordinator():
    cfg = AutoRunConfig(project_id="ptb", budget=Budget(max_wall_clock_s=10))
    assert iterations.claim(cfg, {"round": 0}) is None
    assert iterations.finalize(cfg) is None
