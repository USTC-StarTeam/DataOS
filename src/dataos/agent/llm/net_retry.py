"""Transport-level resilience shared by every ``ClaudeCodeAgent.run`` call.

The Anthropic-compatible endpoints we drive (e.g. DeepSeek's ``/anthropic``) drop
connections mid-response during busy periods, and claude-agent-sdk's ``query()`` has
no auto-reconnect — it just raises. This module classifies which failures are worth
retrying and how long to wait, so the backend can back off and ``resume`` the same
session transparently. Both the main-agent dialog and the execution sub-agents inherit
it for free, because both go through ``ClaudeCodeAgent.run``.

Policy is a WHITELIST: only errors we recognise as transient transport faults are
retried. Anything unknown — including our own :class:`ClaudeCodeError` for genuine
agent failures (bad auth, a real error result) — is surfaced immediately, never
retried into a silent loop.
"""

from __future__ import annotations

# Max reconnect attempts when there is no wall-clock deadline (the main-agent has no
# per-call timeout); sub-agents are additionally bounded by their ``timeout`` budget.
MAX_NET_RETRIES = 6

# Lowercased substrings that mark a transient transport/endpoint fault. Matched against
# both the exception type name and ``str(exc)`` so we catch httpx/anyio errors without
# importing every provider's exception hierarchy.
_NET_ERR_MARKERS: frozenset[str] = frozenset(
    {
        "connection closed",
        "connection reset",
        "connection aborted",
        "connection error",
        "connectionerror",
        "broken pipe",
        "remoteprotocolerror",
        "protocolerror",
        "peer closed connection",
        "server disconnected",
        "incomplete chunked read",
        "read timed out",
        "readtimeout",
        "readerror",
        "connecttimeout",
        "connecterror",
        "econnreset",
        "eof occurred",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "overloaded",
        "too many requests",
        " 429",
        " 500",
        " 502",
        " 503",
        " 504",
    }
)


class ClaudeCodeError(RuntimeError):
    """A genuine (non-transport) failure from the claude_code backend.

    Raised for a real ``is_error`` result, auth failures, etc. Never treated as
    recoverable, so :func:`is_recoverable` returns ``False`` for it and the caller
    sees the error instead of an endless reconnect loop."""


def is_recoverable(exc: BaseException) -> bool:
    """True only for recognised transient transport faults (whitelist)."""
    if isinstance(exc, ClaudeCodeError):
        return False
    hay = f"{type(exc).__name__}\n{exc}".lower()
    return any(marker in hay for marker in _NET_ERR_MARKERS)


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff, capped — ``attempt`` is 1-based (first retry -> 2s)."""
    return float(min(2 ** attempt, 30))
