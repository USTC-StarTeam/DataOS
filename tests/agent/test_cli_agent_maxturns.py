"""Regression: the CLI's max-turns boundary is a RESUMABLE boundary, not a failure.

dataos_agent maps the CLI's ``error_max_turns`` result to ``status="max_steps"``
(carrying the session_id). For a free-form executor experimenter (``output is None``)
the adapter must return that result normally (with the session_id) so the caller
can resume the same session and finish the experiment. Structured single-shot
calls (planner/reviewer, ``output`` set) must still raise, since they need a
complete answer. This bit two live 10h runs before it was fixed.
"""

import dataos_agent
import pytest
from pydantic import BaseModel

import dataos.agent.llm.cli_agent as ca


def _fake_aquery(status: str, session_id: str = "sess-XYZ", error: str | None = None):
    async def q(prompt, **cfg):
        return dataos_agent.Result(text="", status=status, error=error, session_id=session_id)

    return q


def test_executor_resumes_on_max_turns(monkeypatch, tmp_path):
    monkeypatch.setattr(dataos_agent, "aquery", _fake_aquery("max_steps"))
    agent = ca.CliAgent(workspace=str(tmp_path), instructions="x", max_steps=5)
    res = agent.run("do work", output=None)
    # session preserved -> the executor can resume the same session instead of dying
    assert res.session_id == "sess-XYZ"


def test_structured_call_still_raises_on_max_turns(monkeypatch, tmp_path):
    monkeypatch.setattr(dataos_agent, "aquery", _fake_aquery("max_steps"))
    agent = ca.CliAgent(workspace=str(tmp_path), instructions="x", max_steps=5)

    class Out(BaseModel):
        x: int = 0

    with pytest.raises(ca.AgentError):
        agent.run("plan", output=Out)


def test_disconnected_after_healing_raises(monkeypatch, tmp_path):
    """dataos_agent surfaces a disconnected Result only after its reconnect budget is
    exhausted — the adapter must fail the call, not return it as a normal result."""
    monkeypatch.setattr(
        dataos_agent, "aquery", _fake_aquery("disconnected", error="connection reset")
    )
    agent = ca.CliAgent(workspace=str(tmp_path), instructions="x", max_steps=5)
    with pytest.raises(ca.AgentError, match="disconnected"):
        agent.run("do work", output=None)


def test_resume_and_timeout_forwarded(monkeypatch, tmp_path):
    seen = {}

    async def q(prompt, **cfg):
        seen.update(cfg)
        return dataos_agent.Result(text="ok", status="done", session_id="s1")

    monkeypatch.setattr(dataos_agent, "aquery", q)
    agent = ca.CliAgent(
        workspace=str(tmp_path), instructions="x", provider="claude",
        model="some-model", max_steps=5,
    )
    res = agent.run("do work", output=None, resume="sid-9", timeout=42)
    assert res.text == "ok"
    assert seen["provider"] == "claude"
    assert seen["model"] == "some-model"
    assert seen["max_steps"] == 5
    assert seen["timeout"] == 42
    assert seen["extra"] == {"resume": "sid-9"}
