"""run_structured_dialog resilience: an unclassified CLI/transport crash (AgentError)
gets a bounded retry — one flaky CLI exit must not kill a 10h autonomous run."""

import pytest
from pydantic import BaseModel

from dataos.agent.llm.cli_agent import AgentError
from dataos.agent.llm.structured import run_structured_dialog


class _Out(BaseModel):
    answer: str


class _FlakyAgent:
    supports_dialog = True

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def run(self, task, *, output=None, resume=None, timeout=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise AgentError("cli_agent error (error): failed to refresh available models")
        from types import SimpleNamespace

        return SimpleNamespace(text='{"answer": "ok"}', session_id="s1")


def test_agent_error_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("dataos.agent.llm.structured.time.sleep", lambda _s: None)
    agent = _FlakyAgent(failures=2)
    out, sid = run_structured_dialog(agent, "task", _Out)
    assert out.answer == "ok" and sid == "s1"
    assert agent.calls == 3


def test_agent_error_exhaustion_raises(monkeypatch):
    monkeypatch.setattr("dataos.agent.llm.structured.time.sleep", lambda _s: None)
    agent = _FlakyAgent(failures=99)
    with pytest.raises(AgentError):
        run_structured_dialog(agent, "task", _Out)
    assert agent.calls == 1 + 3  # initial + _AGENT_ERROR_RETRIES
