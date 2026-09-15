import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataos.agent import runtime
from dataos.agent import roles as role_runner
from dataos.agent import report as report_schema


def _cfg(ws):
    return SimpleNamespace(
        project_id="p",
        workspace=str(ws),
        target_bench="gsm8k",
        ptb_profile="smoke",
        h100_base_model="/models/qwen-base",
        agent_provider="claude",
        agent_model="sonnet",
        max_steps=1,
        data_limit=20,
        slurm={},
        runtime_config={},
        runtime_config_path=None,
        poll_interval_s=0,
    )


def test_subagent_resumes_same_session_until_report(tmp_path, monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, task, *, output=None, resume=None, timeout=None):
            calls.append({"task": task, "resume": resume, "timeout": timeout})
            if len(calls) == 1:
                return SimpleNamespace(
                    text="submitted train job 123; waiting for completion",
                    session_id="session-123",
                    stop_reason="end_turn",
                )
            match = re.search(r"report\.json is not finished at: (.+)", task)
            assert match is not None
            report_path = Path(match.group(1))
            report_path.write_text(
                json.dumps(
                    {
                        "id": "n0001",
                        "status": "completed",
                        "stages": {
                            "eval": {"score": 95.0, "limit": -1},
                            "train": {"model_path": "/remote/final_model"},
                            "review": {"conclusion": "stub"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(
                text="report written",
                session_id="session-123",
                stop_reason="end_turn",
            )

    monkeypatch.setattr(
        role_runner, "build_agent", lambda *args, **kwargs: FakeAgent(**kwargs)
    )
    monkeypatch.setattr(role_runner.time, "sleep", lambda _seconds: None)

    spec = {"name": "explore-0", "data": {}, "train": {}, "init_model": "/models/parent"}
    _, report = role_runner._run_one(
        _cfg(tmp_path / "ws"), "round-0000", spec, tmp_path / "experiments", timeout=10
    )

    parsed = report_schema.read(report)
    assert parsed.status == "completed"
    assert parsed.score == 95.0
    assert calls[1]["resume"] == "session-123"
    state = json.loads(
        (tmp_path / "experiments" / "explore-0" / "subagent_state.json").read_text()
    )
    assert state["session_id"] == "session-123"
    assert state["status"] == "completed"


def test_execution_host_network_failure_preserves_handle_and_resumes(tmp_path, monkeypatch):
    calls = []
    waits = []

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, task, *, output=None, resume=None, timeout=None):
            calls.append(resume)
            report_path = tmp_path / "experiments" / "explore-0" / "report.json"
            if len(calls) == 1:
                report_path.write_text(
                    json.dumps(
                        {
                            "id": "n0001",
                            "status": "failed",
                            "stages": {
                                "eval": {
                                    "handle": "/remote/eval.handle.json",
                                    "job_id": "123",
                                    "error": "ssh: connect: No route to host",
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                report_path.write_text(
                    json.dumps(
                        {
                            "id": "n0001",
                            "status": "completed",
                            "stages": {
                                "eval": {"score": 0.5, "limit": -1},
                                "review": {"conclusion": "recovered"},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            return SimpleNamespace(
                text="poll complete" if len(calls) > 1 else "network unavailable",
                session_id="session-123",
                stop_reason="end_turn",
            )

    monkeypatch.setattr(role_runner, "build_agent", lambda *args, **kwargs: FakeAgent(**kwargs))
    monkeypatch.setattr(role_runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        role_runner,
        "_wait_for_execution_host",
        lambda _cfg, *, deadline: waits.append(deadline) or True,
    )

    spec = {"name": "explore-0", "data": {}, "train": {}, "init_model": "/models/parent"}
    _, report = role_runner._run_one(
        _cfg(tmp_path / "ws"), "round-0000", spec, tmp_path / "experiments", timeout=10
    )

    assert report_schema.read(report).status == "completed"
    assert len(calls) == 2
    assert calls[1] == "session-123"
    assert waits


def test_real_stage_failure_remains_terminal(tmp_path, monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, task, *, output=None, resume=None, timeout=None):
            calls.append(resume)
            report_path = tmp_path / "experiments" / "explore-0" / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "id": "n0001",
                        "status": "failed",
                        "stages": {
                            "train": {
                                "handle": "/remote/train.handle.json",
                                "job_id": "123",
                                "error": "CUDA out of memory",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(
                text="training failed",
                session_id="session-123",
                stop_reason="end_turn",
            )

    monkeypatch.setattr(role_runner, "build_agent", lambda *args, **kwargs: FakeAgent(**kwargs))
    monkeypatch.setattr(
        role_runner,
        "_wait_for_execution_host",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not wait")),
    )

    spec = {"name": "explore-0", "data": {}, "train": {}, "init_model": "/models/parent"}
    _, report = role_runner._run_one(
        _cfg(tmp_path / "ws"), "round-0000", spec, tmp_path / "experiments", timeout=10
    )

    assert report_schema.read(report).status == "failed"
    assert calls == [None]


def test_runtime_config_injected_into_subagent_prompt(tmp_path):
    cfg = _cfg(tmp_path / "ws")
    cfg.runtime_config = {
        "cluster": {"host": "h100.example", "user": "hcgu", "qos": "gpu_priv"},
        "paths": {
            "deploy_root": "/remote/dataos",
            "runs_base": "/remote/runs",
            "fixed_pool": "/remote/pool",
            "train_env": "/envs/train",
            "eval_env": "/envs/eval",
        },
        "gateway": {"base_url": "https://gateway.example/v1", "key_file": "/keys/gateway"},
    }
    cfg.runtime_config_path = "workspace/p/resolved_config.json"
    spec = {"name": "explore-0", "data": {}, "train": {}, "init_model": "/models/parent"}

    prompt = role_runner._task_prompt(
        cfg=cfg,
        study_id="round-0000",
        spec=spec,
        exp_dir=tmp_path / "experiments" / "explore-0",
        eval_limit=20,
    )

    assert "RUNTIME CONFIG (authoritative" in prompt
    assert '"host": "h100.example"' in prompt
    assert '"runs_base": "/remote/runs"' in prompt
    assert '"run_root": "/remote/runs/p/round-0000/explore-0"' in prompt
    assert '"base_url": "https://gateway.example/v1"' in prompt
    assert "Parent model to continue training from (use this exact path): /models/parent" in prompt
    assert '"parent_model": "/models/parent"' in prompt
    assert runtime._run_root(cfg, "round-0000", "explore-0") == (
        "/remote/runs/p/round-0000/explore-0"
    )


def test_external_task_does_not_require_native_train_or_eval_envs(tmp_path):
    cfg = _cfg(tmp_path / "ws")
    cfg.runtime_config = {
        "kind": "slurm",
        "task": "curationbench",
        "cluster": {
            "host": "h100.example",
            "user": "hcgu",
            "port": 22,
            "key": "/keys/ssh",
            "partition": "gpu",
            "qos": "gpu_priv",
            "gres": "gpu:2",
        },
        "paths": {
            "deploy_root": "/remote/dataos",
            "runs_base": "/remote/runs",
            "fixed_pool": "/remote/pool",
            "train_env": "",
            "eval_env": "",
        },
        "gateway": {"base_url": "https://gateway.example", "key_file": "/keys/gateway"},
    }

    assert runtime._missing_runtime_config_keys(cfg) == []


def test_iteration_budget_has_no_per_experiment_clock(tmp_path):
    from dataos.agent.config import Budget
    from dataos.agent.roles import _exp_timeout

    cfg = _cfg(tmp_path / "ws")
    cfg.budget = Budget(max_rounds=11)

    assert _exp_timeout(cfg, deadline=None) is None


def test_formal_experiment_uses_configured_24h_budget(tmp_path, monkeypatch):
    from dataos.agent.config import Budget
    from dataos.agent.roles import _exp_timeout

    cfg = _cfg(tmp_path / "ws")
    cfg.ptb_profile = "formal"
    cfg.budget = Budget(max_wall_clock_s=24 * 3600)
    monkeypatch.setattr("dataos.agent.roles.time.monotonic", lambda: 100.0)
    assert _exp_timeout(cfg, deadline=None) == 24 * 3600
    assert _exp_timeout(cfg, deadline=100 + 12 * 3600) == 12 * 3600


def test_investigator_uses_configured_baseline_root(tmp_path, monkeypatch):
    from dataos.agent import roles

    cfg = _cfg(tmp_path / "ws")
    cfg.runtime_config = {"paths": {"runs_base": "/configured/runs"}}
    facts = {}
    monkeypatch.setattr(
        "dataos.agent.runtime_compose.render_sections", lambda *a, **k: ("", "", ""),
    )
    monkeypatch.setattr(roles._prompt, "render", lambda name, **kw: facts.update(kw) or "recon")
    monkeypatch.setattr(roles, "system_prompt", lambda *a: "")

    def ready(*args, **kwargs):
        raise RuntimeError("agent construction reached")

    monkeypatch.setattr(roles, "build_agent", ready)
    with pytest.raises(RuntimeError, match="agent construction reached"):
        roles.run_investigator(cfg=cfg, proj=SimpleNamespace(project_dir=tmp_path))
    assert facts["baseline_run_root"] == (
        f"/configured/runs/{cfg.project_id}/round-0000/baseline"
    )


def test_queue_credit_overlap_math():
    """Queue credit = the [submit, wait_end] overlap with THIS invocation. Rows arrive as
    (submit, wait_end) epochs resolved cluster-side: a cancelled-pending job is capped at
    its End (never at now), a still-pending job at now."""
    f = runtime._queue_credit_s
    # cancelled-while-pending: End-capped, finite forever (the P0: no infinite money printer)
    assert f([(100.0, 200.0)], since=0.0) == 100.0
    # still pending at query time: counts submit -> now
    assert f([(100.0, 500.0)], since=0.0) == 400.0
    # entirely before this invocation (resume case): zero
    assert f([(100.0, 200.0)], since=300.0) == 0.0
    # partial overlap: only the in-window part counts
    assert f([(100.0, 400.0)], since=250.0) == 150.0
    # submitted inside the window: whole wait counts
    assert f([(350.0, 400.0)], since=250.0) == 50.0
    # clock-skewed row (end < submit) clamps to zero
    assert f([(400.0, 100.0)], since=0.0) == 0.0
