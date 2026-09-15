"""Transport self-healing: reconnect+resume on recognised transient faults.

Driven by scripted in-process providers (no subprocess, no network). Backoff is
monkeypatched to ~0 so the retry paths run fast; the backoff schedule itself is
unit-tested separately.
"""

import asyncio
import json
import sys

import pytest

from dataos_agent import aquery, query
from dataos_agent._retry import backoff_seconds, is_recoverable, is_recoverable_text
from dataos_agent.agent import _healing_stream
from dataos_agent.providers.base import CliSpec, ProviderSpec, get, register
from dataos_agent.providers.claude import ClaudeStyleSpec
from dataos_agent.types import AgentConfig, Event, Result


class ScriptedSpec(ProviderSpec):
    """Plays a script of attempt outcomes; records the config of every attempt."""

    def __init__(self, name: str, script: list) -> None:
        self.name = name
        self.script = list(script)
        self.calls: list[AgentConfig] = []

    async def stream(self, config: AgentConfig, prompt: str):
        self.calls.append(config)
        action = self.script.pop(0) if self.script else ("ok", "final")
        kind = action[0]
        if kind == "raise":
            raise action[1]
        if kind == "hang":
            yield Event(type="text", text="partial")
            await asyncio.sleep(60)
            return
        if kind == "result":  # ("result", status, error)
            yield Event(type="done", data=Result(
                session_id=f"{self.name}-session", status=action[1], error=action[2]))
            return
        # ("ok", text)
        yield Event(type="text", text=action[1])
        yield Event(type="done", data=Result(
            session_id=f"{self.name}-session", text=action[1], status="done"))


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr("dataos_agent.agent.backoff_seconds", lambda attempt: 0.001)


def make_spec(name: str, script: list) -> ScriptedSpec:
    spec = ScriptedSpec(name, script)
    register(spec)
    return spec


# --- reconnect + resume ---------------------------------------------------

async def test_error_result_reconnects_with_resume():
    spec = make_spec("s1", [
        ("result", "error", "Connection closed mid-response"),
        ("result", "error", "Connection closed mid-response"),
        ("ok", "survived"),
    ])
    r = await aquery("hi", provider="s1")
    assert r.status == "done" and r.text == "survived"
    assert len(spec.calls) == 3
    # the reconnects resumed the session learned before the drop
    assert spec.calls[1].extra.get("resume") == "s1-session"
    assert spec.calls[2].extra.get("resume") == "s1-session"


async def test_raised_transport_fault_reconnects():
    spec = make_spec("s2", [
        ("raise", RuntimeError("Connection reset by peer")),
        ("ok", "back"),
    ])
    r = await aquery("hi", provider="s2")
    assert r.status == "done" and r.text == "back"
    assert len(spec.calls) == 2


async def test_claude_generic_server_error_resumes_same_session():
    spec = make_spec("claude_server", [
        ("result", "error", "API Error: Server error。 请求 ID：example"),
        ("ok", "recovered"),
    ])
    result = await aquery("continue", provider=spec.name)
    assert result.text == "recovered"
    assert spec.calls[1].extra["resume"] == "claude_server-session"


@pytest.mark.parametrize("reason", [
    "authentication failed", "invalid api key", "insufficient_quota",
    "credit balance too low", "invalid_request_error", "HTTP 401", "HTTP 403",
])
def test_fatal_errors_are_not_hidden_by_server_error_text(reason):
    assert not is_recoverable_text(f"API Error: Server error: {reason}")


async def test_disconnected_result_reconnects():
    spec = make_spec("s3", [("result", "disconnected", "eof occurred"), ("ok", "back")])
    r = await aquery("hi", provider="s3")
    assert r.status == "done"
    assert len(spec.calls) == 2


async def test_caller_supplied_resume_is_kept():
    spec = make_spec("s4", [("ok", "hi")])
    await aquery("hi", provider="s4", extra={"resume": "caller-sid"})
    assert spec.calls[0].extra.get("resume") == "caller-sid"


# --- budget exhaustion ----------------------------------------------------

async def test_net_retries_cap_surfaces_disconnected():
    spec = make_spec("s5", [("result", "error", "connection reset")] * 10)
    r = await aquery("hi", provider="s5", net_retries=2)
    assert r.status == "disconnected"
    assert len(spec.calls) == 3  # 1 initial + 2 reconnects


async def test_net_retries_zero_disables_healing():
    spec = make_spec("s6", [("result", "disconnected", "eof occurred")])
    r = await aquery("hi", provider="s6", net_retries=0)
    assert r.status == "disconnected"
    assert len(spec.calls) == 1


async def test_non_recoverable_error_result_not_retried():
    spec = make_spec("s7", [("result", "error", "authentication failed")] * 5)
    r = await aquery("hi", provider="s7")
    assert r.status == "error" and "authentication" in r.error
    assert len(spec.calls) == 1


async def test_non_recoverable_exception_raises():
    make_spec("s8", [("raise", ValueError("bad argv"))])
    with pytest.raises(ValueError):
        await aquery("hi", provider="s8")


# --- wall-clock timeout ----------------------------------------------------

async def test_timeout_covers_reconnects():
    # enough scripted failures that only the wall clock (not the script) can end the loop
    make_spec("s9", [("result", "error", "connection reset")] * 100_000)
    with pytest.raises(TimeoutError):
        await aquery("hi", provider="s9", timeout=0.05)


async def test_timeout_aborts_a_hung_attempt():
    make_spec("s10", [("hang",)])
    with pytest.raises(TimeoutError):
        await aquery("hi", provider="s10", timeout=0.05)


# --- the stream contract ----------------------------------------------------

async def test_failed_attempts_done_events_are_swallowed():
    """Consumers see exactly one done event, for the final attempt."""
    spec = make_spec("s11", [("result", "disconnected", "eof occurred"), ("ok", "back")])
    events = []
    async for ev in _healing_stream(spec, AgentConfig(provider="s11"), "hi"):
        events.append(ev)
    assert [e.type for e in events].count("done") == 1
    assert events[-1].data.text == "back"


def test_sync_query_heals():
    make_spec("s12", [("result", "disconnected", "eof occurred"), ("ok", "back")])
    r = query("hi", provider="s12")
    assert r.status == "done" and r.text == "back"


async def test_sync_query_rejects_a_running_loop():
    with pytest.raises(RuntimeError, match="aquery"):
        query("hi", provider="fake")


# --- stall watchdog (idle_timeout) ------------------------------------------

_HANG_SCRIPT = (
    "import json,time;"
    "print(json.dumps({'type':'session','id':'stall-sid'}),flush=True);"
    "print(json.dumps({'type':'text','text':'partial'}),flush=True);"
    "time.sleep(30)"
)
_OK_SCRIPT = "import json;print(json.dumps({'type':'text','text':'back'}),flush=True)"


class StallCliSpec(CliSpec):
    """A real-subprocess CLI: the fresh attempt prints a session id + one event then
    falls silent forever; the resumed attempt answers and exits. Exercises the
    idle_timeout watchdog through the actual run_cli machinery."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.bin = sys.executable
        self.calls: list[AgentConfig] = []

    def build_argv(self, config: AgentConfig, prompt: str) -> list[str]:
        self.calls.append(config)
        script = _OK_SCRIPT if config.extra.get("resume") else _HANG_SCRIPT
        return ["-c", script]

    def parse_line(self, raw: dict, acc: Result):
        if raw.get("type") == "session":
            acc.session_id = raw["id"]
        elif raw.get("type") == "text":
            acc.text += raw["text"]
            yield Event(type="text", text=raw["text"])


async def test_idle_stall_reconnects_with_resume():
    spec = StallCliSpec("stall1")
    register(spec)
    r = await aquery("hi", provider="stall1", idle_timeout=0.2)
    assert r.status == "done" and r.text == "back"
    assert len(spec.calls) == 2
    # the reconnect resumed the session id rescued before the stall
    assert spec.calls[1].extra.get("resume") == "stall-sid"


async def test_idle_stall_without_retries_surfaces_disconnected():
    spec = StallCliSpec("stall2")
    register(spec)
    r = await aquery("hi", provider="stall2", idle_timeout=0.2, net_retries=0)
    assert r.status == "disconnected" and "stalled" in (r.error or "")
    assert len(spec.calls) == 1


# --- policy units -------------------------------------------------------------

def test_backoff_schedule():
    assert [backoff_seconds(i) for i in (1, 2, 3, 4, 5, 6)] == [2, 4, 8, 16, 30, 30]


def test_marker_classification():
    assert is_recoverable_text("Connection closed mid-response")
    assert is_recoverable_text("HTTP 503 Service Unavailable")
    assert not is_recoverable_text("authentication failed")
    assert is_recoverable(RuntimeError("broken pipe"))
    assert not is_recoverable(ValueError("bad argv"))
    # codex CLI internal hangs (the gpqamain-bare run died unrecovered on these)
    assert is_recoverable_text("failed to refresh available models: timeout waiting "
                               "for child process to exit")


# --- claude mapper: max_turns is a benign boundary -----------------------------

def test_claude_max_turns_maps_to_max_steps():
    spec = get("claude")
    assert isinstance(spec, ClaudeStyleSpec)
    acc = Result(session_id="sid")
    list(spec.parse_line({
        "type": "result", "is_error": True, "subtype": "error_max_turns",
        "result": "partial answer", "session_id": "sid",
    }, acc))
    assert acc.status == "max_steps"
    assert acc.text == "partial answer"


def test_transcript_tee(tmp_path):
    """transcript=<path> tees every neutral event (incl. across a reconnect) as JSONL."""
    log = tmp_path / "agent.jsonl"
    make_spec("s13", [("result", "disconnected", "eof occurred"), ("ok", "back")])
    r = query("hi", provider="s13", transcript=str(log))
    assert r.status == "done"
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    types = [e["type"] for e in lines]
    assert "text" in types and "done" in types
    assert all("ts" in e for e in lines)
    assert lines[-1]["type"] == "done" and lines[-1]["data"]["text"] == "back"
    # exactly one done (the failed attempt's was swallowed, per the stream contract)
    assert types.count("done") == 1


# --- pipe hygiene -------------------------------------------------------------

_STDERR_FLOOD_SCRIPT = (
    "import json,sys;"
    "sys.stderr.write('x' * 200_000);sys.stderr.flush();"
    "print(json.dumps({'type':'text','text':'survived'}),flush=True)"
)


class FloodCliSpec(CliSpec):
    """Real subprocess that floods stderr with 200KB before answering — a PIPE read only
    after process exit deadlocks at the 64KB pipe buffer; the concurrent drain must not."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.bin = sys.executable

    def build_argv(self, config: AgentConfig, prompt: str) -> list[str]:
        return ["-c", _STDERR_FLOOD_SCRIPT]

    def parse_line(self, raw: dict, acc: Result):
        if raw.get("type") == "text":
            acc.text += raw["text"]
            yield Event(type="text", text=raw["text"])


async def test_stderr_flood_does_not_deadlock():
    spec = FloodCliSpec("flood1")
    register(spec)
    r = await asyncio.wait_for(aquery("hi", provider="flood1"), timeout=15)
    assert r.status == "done" and r.text == "survived"


def test_limit_overrun_is_recoverable():
    """A single stream line beyond the 16MiB read limit heals via kill+resume."""
    exc = asyncio.LimitOverrunError("Separator is not found, and chunk exceed the limit", 100)
    assert is_recoverable(exc)
