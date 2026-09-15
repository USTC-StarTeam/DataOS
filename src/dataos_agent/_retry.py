"""Transport-level self-healing policy: which failures are worth retrying, and how
long to wait before reconnecting.

The Anthropic-compatible endpoints these CLIs drive drop connections mid-response
during busy periods, and the CLIs themselves have no auto-reconnect — the stream
just dies. This module classifies recognised transient transport faults (a
WHITELIST — anything unknown surfaces immediately, never retried into a silent
loop) so the healing wrapper in ``agent.py`` can back off and ``resume`` the same
session transparently.

Ported from DataOS's battle-tested ``llm/net_retry.py`` (the brain's sub-agents
run 10h experiments over these endpoints); keep the markers in sync with it.
"""

from __future__ import annotations

# Reconnect cap when there is no wall-clock timeout; with a timeout the budget
# bounds the retries instead.
DEFAULT_NET_RETRIES = 6

# Lowercased substrings that mark a transient transport/endpoint fault. Matched
# against both the exception type name and ``str(exc)`` / the CLI's error text,
# so we catch httpx/anyio errors and CLI-rendered messages without importing
# every provider's exception hierarchy.
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
        "api error: server error",
        "internal server error",
        "too many requests",
        # codex CLI internal hang (models_manager refresh child never exits) — the CLI
        # prints it to stderr and dies; it is a transport-side fault, resume-safe
        "failed to refresh available models",
        "timeout waiting for child process",
        " 429",
        " 500",
        " 502",
        " 503",
        " 504",
        # a single stream line beyond the 16MiB read limit (codex aggregated_output of a
        # huge command log): killing + resuming escapes it — the giant line is not
        # re-streamed on resume
        "limitoverrunerror",
        "separator is not found",
    }
)


def is_recoverable_text(text: str) -> bool:
    """True if a raw error STRING looks like a transient transport fault.

    Used both for raised exceptions (path A) and for error text the CLI renders
    into its result event instead of raising (path B, e.g. "Connection closed
    mid-response")."""
    hay = str(text).lower()
    if any(marker in hay for marker in (
        "authentication failed", "invalid api key", "insufficient_quota",
        "credit balance", "invalid_request_error", " 401", " 403",
    )):
        return False
    return any(marker in hay for marker in _NET_ERR_MARKERS)


def is_recoverable(exc: BaseException) -> bool:
    """True only for recognised transient transport faults (whitelist)."""
    return is_recoverable_text(f"{type(exc).__name__}\n{exc}")


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff, capped — ``attempt`` is 1-based (first retry -> 2s)."""
    return float(min(2**attempt, 30))
