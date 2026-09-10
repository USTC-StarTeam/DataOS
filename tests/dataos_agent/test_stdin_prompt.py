import hashlib
import sys

import pytest

from dataos_agent._cli import run_cli
from dataos_agent.providers.codex import CodexSpec
from dataos_agent.types import AgentConfig, Event


@pytest.mark.parametrize("resume", [False, True])
async def test_large_codex_prompt_reaches_stdin_exactly(monkeypatch, resume):
    from dataos_agent.providers import base

    prompt = "中文 context\n" * 50000
    cfg = AgentConfig(provider="codex", instructions="role", model="gpt-5.6-sol", effort="max",
                      extra={"resume": "session"} if resume else {})

    async def capture(bin, argv, **kwargs):
        assert argv[-1] == "-"
        assert max(len(arg) for arg in argv) < 1000
        assert kwargs["stdin_text"] == (prompt if resume else "role\n\n" + prompt)
        assert "model_reasoning_effort=max" in argv
        yield Event(type="text", text="ok")

    monkeypatch.setattr(base, "run_cli", capture)
    assert [ev.text async for ev in CodexSpec().stream(cfg, prompt)] == ["ok"]


async def test_subprocess_stdin_is_complete_and_output_does_not_deadlock():
    payload = "中文\n" * 300000
    child = (
        "import sys,json,hashlib; "
        "sys.stderr.write('x'*100000); sys.stderr.flush(); "
        "data=sys.stdin.buffer.read(); "
        "print(json.dumps({'digest':hashlib.sha256(data).hexdigest()}))"
    )

    def parse(obj, acc):
        acc.text = obj["digest"]
        return []

    events = [e async for e in run_cli(sys.executable, ["-c", child], stdin_text=payload,
                                      parse_line=parse, idle_timeout=10)]
    assert events[-1].data.status == "done"
    assert events[-1].data.text == hashlib.sha256(payload.encode()).hexdigest()


async def test_nonzero_exit_retains_exit_code_and_tail_not_only_warning():
    child = "import sys; sys.stderr.write('WARNING stale dirs\\n'+'x'*3000+'REAL ERROR'); sys.exit(7)"
    events = [e async for e in run_cli(sys.executable, ["-c", child], parse_line=lambda *_: [])]
    result = events[-1].data
    assert result.status == "error"
    assert "exit code 7" in result.error
    assert result.error.endswith("REAL ERROR")
