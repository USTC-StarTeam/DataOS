"""Typed errors. All inherit DataOSAgentError so callers can catch one base."""

from __future__ import annotations


class DataOSAgentError(Exception):
    """Base for everything this package raises."""


class UnknownProvider(DataOSAgentError):
    """Asked for a provider that isn't registered."""


class CLINotFound(DataOSAgentError):
    """The backend CLI binary (claude/kimi/qwen/...) isn't on PATH."""


class ProviderError(DataOSAgentError):
    """The backend CLI ran but failed (non-zero exit / error event)."""


class DisconnectError(DataOSAgentError):
    """Transient transport failure (process died / stream cut). Recoverable — retry."""


class OutputParseError(DataOSAgentError):
    """Asked for structured output (schema=) but the agent's text didn't yield a
    valid instance of it."""
