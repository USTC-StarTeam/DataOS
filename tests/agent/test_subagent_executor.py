import json
import re
from pathlib import Path
from types import SimpleNamespace

from dataos.agent import subagent_executor as subagent_exec


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
            match = re.search(r"report\.json is still missing at: (.+)", task)
            assert match is not None
            report_path = Path(match.group(1))
            report_path.write_text(
                json.dumps(
                    {
                        "name": "explore-0",
                        "status": "completed",
                        "score": 95.0,
                        "model": {"path": "/remote/final_model"},
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
        subagent_exec, "build_agent", lambda *args, **kwargs: FakeAgent(**kwargs)
    )
    monkeypatch.setattr(subagent_exec.time, "sleep", lambda _seconds: None)

    spec = {"name": "explore-0", "data": {}, "train": {}, "init_model": "/models/parent"}
    _, report = subagent_exec._run_one(
        _cfg(tmp_path / "ws"), "round-0000", spec, tmp_path / "experiments", timeout=10
    )

    assert report["status"] == "completed"
    assert report["score"] == 95.0
    assert calls[1]["resume"] == "session-123"
    state = json.loads(
        (tmp_path / "experiments" / "explore-0" / "subagent_state.json").read_text()
    )
    assert state["session_id"] == "session-123"
    assert state["status"] == "completed"


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

    prompt = subagent_exec._task_prompt(
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
    assert subagent_exec._run_root(cfg, "round-0000", "explore-0") == (
        "/remote/runs/p/round-0000/explore-0"
    )
