"""Regression: Claude Code ``max_turns`` is a RESUMABLE boundary, not a failure.

The claude-agent-sdk emits the ``is_error`` result (which carries the session_id)
and THEN raises on the CLI's non-zero exit. For a free-form executor sub-agent
(``output is None``) the wrapper must preserve that result and return it (with the
session_id) so the caller can resume the same session and finish the experiment.
Structured single-shot calls (planner/reviewer, ``output`` set) must still raise,
since they need a complete answer. This bit two live 10h runs before it was fixed.
"""

from types import SimpleNamespace

import claude_agent_sdk
import pytest
from pydantic import BaseModel

import dataos.agent.llm.claude_code as cc


def _fake_max_turns_query():
    """Mimic the SDK: yield the is_error max_turns result, then raise on exit."""

    async def q(prompt, options):
        yield SimpleNamespace(
            is_error=True,
            subtype="error_max_turns",
            errors=["Reached maximum number of turns (150)"],
            result=None,
            session_id="sess-XYZ",
        )
        raise RuntimeError(
            "Claude Code returned an error result: Reached maximum number of turns (150)"
        )

    return q


def test_executor_resumes_on_max_turns(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_agent_sdk, "query", _fake_max_turns_query())
    agent = cc.ClaudeCodeAgent(workspace=str(tmp_path), instructions="x", max_steps=5)
    res = agent.run("do work", output=None)
    # session preserved -> the executor can resume the same session instead of dying
    assert res.session_id == "sess-XYZ"


def test_structured_call_still_raises_on_max_turns(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_agent_sdk, "query", _fake_max_turns_query())
    agent = cc.ClaudeCodeAgent(workspace=str(tmp_path), instructions="x", max_steps=5)

    class Out(BaseModel):
        x: int = 0

    with pytest.raises(Exception):
        agent.run("plan", output=Out)
