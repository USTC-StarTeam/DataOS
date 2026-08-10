"""dataos_agent — one interface over many code-agent CLIs (claude / kimi / qwen ...).

    from dataos_agent import query
    print(query("What is 2 + 2?", provider="claude"))
"""

from __future__ import annotations

from . import providers  # noqa: F401  — self-registers built-in providers
from .agent import Client, aquery, query
from ._home import context_filename
from .errors import (
    CLINotFound,
    DataOSAgentError,
    DisconnectError,
    OutputParseError,
    ProviderError,
    UnknownProvider,
)
from .providers.base import providers as list_providers
from .providers.base import register
from .types import AgentConfig, Event, Result, Status, Step, Usage

__version__ = "0.3.0"

__all__ = [
    # entry points
    "query",
    "aquery",
    "Client",
    # types
    "AgentConfig",
    "Result",
    "Event",
    "Usage",
    "Step",
    "Status",
    # errors
    "DataOSAgentError",
    "OutputParseError",
    "ProviderError",
    "DisconnectError",
    "CLINotFound",
    "UnknownProvider",
    # provider knowledge (the caller builds the file; this only names it)
    "context_filename",
    # extension
    "register",
    "list_providers",
    "__version__",
]
