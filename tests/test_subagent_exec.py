import json
import re
from pathlib import Path
from types import SimpleNamespace

from dataos_auto import subagent_exec


def _cfg(ws):
    return SimpleNamespace(
        project_id="p",
        workspace=str(ws),
        target_bench="gsm8k",
        ptb_profile="smoke",
        h100_base_model="/models/qwen-base",
        claude_code_model="sonnet",
        max_steps=1,
        data_limit=20,
        slurm={},
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

    monkeypatch.setattr(subagent_exec, "ClaudeCodeAgent", FakeAgent)
    monkeypatch.setattr(subagent_exec.time, "sleep", lambda _seconds: None)

    spec = {"name": "explore-0", "data": {}, "train": {}}
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
