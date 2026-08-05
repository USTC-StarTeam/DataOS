"""The two entry points.

  query(prompt, provider=..., schema=...)   -> Result          (one-shot, sync)
  async with Client(provider=...) as c:      ...                (interactive, streaming)

Both reduce a backend's neutral Event stream into a single Result; ``schema=``
additionally validates the final text into ``result.data``.

Every stream goes through :func:`_healing_stream`: on a recognised transient
transport fault (the endpoint drops the connection mid-response) it backs off
and reconnects with ``resume=<recovered session id>``, so a long-running agent
survives flaky endpoints transparently. Bounded by ``config.timeout`` (wall
clock, covers reconnects) when set, and by ``config.net_retries`` otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from ._output import parse_output
from ._retry import backoff_seconds, is_recoverable, is_recoverable_text
from .providers.base import get
from .types import AgentConfig, Event, Result

log = logging.getLogger(__name__)


def _tee(config: AgentConfig, ev: Event) -> None:
    """Append one neutral Event to the transcript file (provider-neutral run record).
    Best-effort: a transcript write failure must never break the run."""
    if not config.transcript:
        return
    try:
        p = Path(config.transcript)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **dataclasses.asdict(ev)},
                               ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("transcript write failed: %s", e)


async def _healing_stream(spec, config: AgentConfig, prompt: str) -> AsyncIterator[Event]:
    """Yield the run's Events with transport self-healing.

    A transient fault shows up in one of three shapes: the stream raises it
    (path A), the CLI renders it into an error/disconnected result event
    (path B), or the stream ends with no result at all. On any of these we
    discard the attempt's partial result, sleep ``backoff_seconds(attempt)``
    (capped by the remaining wall-clock budget), and reconnect with the
    session id recovered before the drop. ``done`` is yielded exactly once,
    for the final attempt; when the reconnect budget is exhausted the failure
    is surfaced as a ``disconnected`` Result rather than an exception."""
    deadline = (time.monotonic() + config.timeout) if config.timeout else None
    resume = config.extra.get("resume")
    attempt = 0
    while True:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f"dataos_agent timed out after {config.timeout}s")
        turn_cfg = config
        if resume:
            turn_cfg = config.model_copy(update={"extra": {**config.extra, "resume": resume}})

        last: Optional[Result] = None
        reason: Optional[str] = None
        agen = spec.stream(turn_cfg, prompt)
        try:
            ctx = asyncio.timeout(remaining) if remaining is not None else contextlib.nullcontext()
            async with ctx:
                async for ev in agen:
                    if ev.type == "done":
                        last = ev.data
                        resume = ev.data.session_id or resume
                        continue  # the final attempt's done is yielded below
                    _tee(config, ev)
                    yield ev
        except TimeoutError:
            # the wall-clock budget is spent — not a transport fault; don't retry.
            raise TimeoutError(f"dataos_agent timed out after {config.timeout}s") from None
        except Exception as e:  # noqa: BLE001 — path A: the stream raised the drop
            if not is_recoverable(e):
                raise
            reason = f"{type(e).__name__}: {e}"
        finally:
            with contextlib.suppress(Exception):
                await agen.aclose()

        if reason is None and last is not None:
            if last.status == "disconnected":
                reason = last.error or "disconnected"
            elif last.status == "error" and last.error and is_recoverable_text(last.error):
                reason = last.error  # path B: the CLI rendered the drop into its result
        if reason is None:
            if last is None:
                last = Result(status="error", error="backend produced no result event")
            done = Event(type="done", data=last)
            _tee(config, done)
            yield done
            return

        attempt += 1
        if config.net_retries <= 0 or (deadline is None and attempt > config.net_retries):
            # reconnect budget exhausted: surface the failure, never loop silently
            if last is None:
                last = Result(session_id=resume)
            last.status = "disconnected"
            last.error = (last.error or reason)[:2000]
            done = Event(type="done", data=last)
            _tee(config, done)
            yield done
            return
        wait = backoff_seconds(attempt)
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - time.monotonic()))
        log.warning(
            "transport fault (%s); reconnect #%d (resume=%s) in %.0fs",
            reason, attempt, resume, wait,
        )
        await asyncio.sleep(wait)


async def _drain(spec, config: AgentConfig, prompt: str, schema) -> Result:
    result: Optional[Result] = None
    async for ev in _healing_stream(spec, config, prompt):
        if ev.type == "done":
            result = ev.data
    assert result is not None  # _healing_stream always yields exactly one done
    if schema is not None and result.status == "done":
        result.data = parse_output(result.text, schema)   # raises OutputParseError
    return result


async def aquery(prompt: str, *, provider: str, schema: type | None = None, **cfg: Any) -> Result:
    """Async one-shot: run to completion, return the reduced Result."""
    config = AgentConfig(provider=provider, **cfg)
    return await _drain(get(provider), config, prompt, schema)


def query(prompt: str, *, provider: str, schema: type | None = None, **cfg: Any) -> Result:
    """Sync one-shot. ``print(query("hi", provider="claude"))`` just works."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "query() cannot be called from a running event loop; "
            "use `await aquery(...)` instead"
        ) from None
    return asyncio.run(aquery(prompt, provider=provider, schema=schema, **cfg))


class Client:
    """Interactive / multi-turn. Stream neutral Events, keep the session across
    turns (via the backend's resume), and interrupt a run in flight."""

    def __init__(self, *, provider: str, schema: type | None = None, **cfg: Any) -> None:
        self.config = AgentConfig(provider=provider, **cfg)
        self.schema = schema
        self._spec = get(provider)
        self.result: Optional[Result] = None
        self.session_id: Optional[str] = None
        self._active: Optional[AsyncIterator[Event]] = None

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        await self.interrupt()
        return False

    def _turn_config(self) -> AgentConfig:
        if not self.session_id:
            return self.config
        return self.config.model_copy(
            update={"extra": {**self.config.extra, "resume": self.session_id}}
        )

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Yield this turn's neutral Events; updates ``result`` / ``session_id`` at ``done``."""
        gen = _healing_stream(self._spec, self._turn_config(), prompt)
        self._active = gen
        try:
            async for ev in gen:
                if ev.type == "done":
                    self.result = ev.data
                    self.session_id = ev.data.session_id or self.session_id
                    if self.schema and ev.data.status == "done":
                        ev.data.data = parse_output(ev.data.text, self.schema)
                yield ev
        finally:
            self._active = None

    async def run(self, prompt: str) -> Result:
        """Run one turn to completion (no streaming), return its Result."""
        async for _ in self.stream(prompt):
            pass
        assert self.result is not None
        return self.result

    async def interrupt(self) -> None:
        """Best-effort: close the in-flight stream, which tears down the subprocess."""
        gen = self._active
        self._active = None
        if gen is not None:
            aclose = getattr(gen, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
