"""Offline: replay recorded stream-json fixtures through each provider's mapper
and assert the neutral Result. Deterministic — no CLI, no network."""

import json
import pathlib

from dataos_agent.providers.claude import ClaudeSpec
from dataos_agent.providers.kimi import KimiSpec
from dataos_agent.providers.qwen import QwenSpec
from dataos_agent.types import Result

FIX = pathlib.Path(__file__).parent / "fixtures"


def replay(spec, name, stderr=""):
    acc = Result()
    events = []
    for line in (FIX / f"{name}.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        for ev in spec.parse_line(json.loads(line), acc):
            events.append(ev)
    spec.finalize(acc, stderr)
    return events, acc


def test_claude_mapper():
    events, acc = replay(ClaudeSpec(), "claude")
    assert acc.text == "OK"
    assert acc.status == "done"
    assert acc.session_id  # set from system/result
    assert acc.usage.output_tokens == 4
    assert acc.usage.cost_usd > 0
    assert any(e.type == "text" for e in events)


def test_qwen_mapper_same_shape():
    # qwen is a Claude-Code-derived CLI -> same stream-json shape as claude.
    events, acc = replay(QwenSpec(), "qwen")
    assert acc.text == "OK"
    assert acc.status == "done"
    assert acc.session_id
    assert acc.usage.output_tokens == 496
    assert any(e.type == "text" for e in events)


def test_kimi_mapper_and_session_from_stderr():
    events, acc = replay(
        KimiSpec(), "kimi",
        stderr="To resume this session: kimi -r 6a382934-4f84-4916-99e2-e244a5ba6ca1",
    )
    assert acc.text == "OK"
    assert acc.session_id == "6a382934-4f84-4916-99e2-e244a5ba6ca1"
    assert any(e.type == "text" for e in events)


def test_kimi_mapper_tool_calls():
    """Real recorded tools=on stream: OpenAI-style tool_calls + role:'tool' lines."""
    events, acc = replay(KimiSpec(), "kimi_tools")
    assert acc.text == "DONE"
    assert len(acc.steps) == 1
    step = acc.steps[0]
    assert step.tool == "WriteFile"
    assert step.args == {"path": "/tmp/kimi-raw-test/probe.txt", "content": "kimi probe"}
    assert "File successfully overwritten" in step.result  # backfilled from role:'tool'
    assert any(e.type == "tool_use" and e.name == "WriteFile" for e in events)
    assert any(e.type == "tool_result" for e in events)


def test_claude_argv_maps_effort():
    from dataos_agent.types import AgentConfig

    argv = ClaudeSpec().build_argv(
        AgentConfig(provider="claude", model="m1", effort="medium"), "hi")
    assert "--effort" in argv and argv[argv.index("--effort") + 1] == "medium"
    argv2 = ClaudeSpec().build_argv(AgentConfig(provider="claude"), "hi")
    assert "--effort" not in argv2


def test_codex_mapper():
    from dataos_agent.providers.codex import CodexSpec

    events, acc = replay(CodexSpec(), "codex")
    assert acc.text == "OK"
    assert acc.session_id == "019fc3fd-1609-7730-b230-a964fd77f832"
    assert acc.usage.input_tokens == 15081 and acc.usage.output_tokens == 5


def test_codex_mapper_tools():
    from dataos_agent.providers.codex import CodexSpec

    events, acc = replay(CodexSpec(), "codex_tools")
    assert acc.text == "DONE"
    assert len(acc.steps) == 1
    assert acc.steps[0].tool == "Bash" and acc.steps[0].result == "x"
    assert any(e.type == "tool_use" and e.name == "Bash" for e in events)
    assert any(e.type == "tool_result" for e in events)


def test_codex_argv():
    from dataos_agent.providers.codex import CodexSpec
    from dataos_agent.types import AgentConfig

    argv = CodexSpec().build_argv(
        AgentConfig(provider="codex", model="gpt-5.6-sol", effort="medium",
                    workspace="/ws", extra={"resume": "tid-1"}), "hi")
    assert argv[:3] == ["exec", "resume", "tid-1"]
    assert "-m" in argv and "gpt-5.6-sol" in argv
    assert any("model_reasoning_effort=medium" in a for a in argv)
    assert argv[-1] == "hi"  # prompt last


def test_kimi_new_cli_shape():
    events, acc = replay(KimiSpec(), "kimi_new")
    assert acc.text == "DONE"
    assert acc.session_id == "session_eceaab70-8dc7-48c1-9b2c-aa9e6c78999a"
    assert len(acc.steps) == 1 and acc.steps[0].tool == "Write"
    assert acc.steps[0].result == "Wrote 1 bytes to n.txt"
