"""The one shared machine every CLI backend uses: spawn ``<bin> ... --output-format
stream-json``, read stdout line-by-line as JSON events, hand each to the provider's
``parse_line`` (which mutates a running Result and yields neutral Events), then
finalise status from the exit code. No provider-specific logic lives here."""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import AsyncIterator, Callable, Iterable

from ._retry import is_recoverable_text
from .errors import CLINotFound
from .types import Event, Result

# parse_line(raw_json_obj, running_result) -> yields neutral Events, mutates the Result.
ParseLine = Callable[[dict, Result], Iterable[Event]]


async def run_cli(
    bin: str,
    argv: list[str],
    *,
    cwd: str | None = None,
    env: dict | None = None,
    parse_line: ParseLine,
    finalize: Callable[[Result, str], None] | None = None,
    idle_timeout: float | None = None,
) -> AsyncIterator[Event]:
    exe = shutil.which(bin) or (bin if bin.startswith("/") else None)
    if exe is None:
        raise CLINotFound(f"CLI not found on PATH: {bin!r}")

    proc = await asyncio.create_subprocess_exec(
        exe, *argv,
        stdin=asyncio.subprocess.DEVNULL,   # headless: never block waiting on stdin
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        # codex emits huge single-line JSON events (aggregated command output);
        # the default 64KiB stream limit kills the read with LimitOverrunError
        limit=16 * 1024 * 1024,
    )
    acc = Result()

    async def _read() -> AsyncIterator[Event]:
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON chatter — skip
            for ev in parse_line(obj, acc):
                yield ev

    try:
        if idle_timeout is not None:
            # stall watchdog: bound EACH event's wait; on expiry kill + mark the
            # attempt disconnected so the healing wrapper resumes the session —
            # a silent stream is a transport fault, not a terminal timeout.
            it = _read().__aiter__()
            while True:
                try:
                    ev = await asyncio.wait_for(it.__anext__(), timeout=idle_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    proc.kill()
                    acc.status = "disconnected"
                    acc.error = (
                        f"stalled: no output for {idle_timeout}s "
                        "(killed; session resumable)"
                    )
                    break
                else:
                    yield ev
        else:
            async for ev in _read():
                yield ev
    finally:
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            rc = await proc.wait()
        stderr = b""
        if proc.stderr is not None:
            try:
                stderr = await proc.stderr.read()
            except Exception:
                stderr = b""
        stderr_text = stderr.decode("utf-8", "replace")
        if acc.status == "done" and rc not in (0, None):
            # non-zero exit that the stream didn't already explain
            acc.status = "error"
            acc.error = (stderr_text.strip() or f"exit code {rc}")[:2000]
        if acc.status == "error" and acc.error and is_recoverable_text(acc.error):
            # recognised transient transport fault — the healing wrapper reconnects+resumes
            acc.status = "disconnected"
        if finalize is not None:
            finalize(acc, stderr_text)   # provider can enrich from stderr (e.g. session id)

    yield Event(type="done", data=acc)
