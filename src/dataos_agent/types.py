"""The neutral, provider-agnostic surface: config in, Result out, Event as the
one streaming type. No backend's native types leak through these."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

# Why a run ended. Richer than a bool so the brain can react (disconnected -> retry).
Status = Literal["done", "max_steps", "timeout", "disconnected", "error", "refused"]

# The one streaming event type. Every backend's stream maps onto these four.
EventType = Literal["text", "tool_use", "tool_result", "done"]


class AgentConfig(BaseModel):
    """Validated, serializable config — the knobs common to every code-agent CLI.
    Backend-specific flags go in ``extra`` (pass-through). ``extra='forbid'``
    catches typos early."""

    model_config = ConfigDict(extra="forbid")

    provider: str                                   # claude | kimi | qwen | fake | ...
    model: Optional[str] = None                     # None -> the CLI's default model
    instructions: Optional[str] = None              # system prompt / append-system-prompt
    effort: Optional[str] = None                    # thinking level (claude: low/medium/high/xhigh/max)
    tools: bool | list[str] = True                  # native-tool policy: True all / [..] allowlist / False none
    workspace: Optional[str] = None                 # cwd / work-dir; files land here
    max_steps: Optional[int] = None                 # step cap (max-turns)
    permission: str = "auto"                        # auto (auto-approve) | ask | deny
    timeout: Optional[float] = None                 # wall-clock budget for the whole call,
                                                    # COVERS transport reconnects; expiry -> TimeoutError
    idle_timeout: Optional[float] = None            # stall watchdog: kill the CLI after Ns of
                                                    # stdout silence -> "disconnected" -> the healing
                                                    # wrapper resumes the session (not the wall clock)
    net_retries: int = 6                            # transport reconnect cap (0 disables self-healing);
                                                    # bounds retries only when timeout is unset
    transcript: Optional[str] = None                # append every neutral Event as a JSON line
                                                    # to this file (provider-neutral run record)
    extra: dict[str, Any] = {}                       # backend-specific escape hatch


@dataclass
class Usage:
    """Token accounting (+ optional cost). Accumulated across a run's turns."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, *, input: int = 0, output: int = 0, total: int | None = None, cost: float = 0.0) -> None:
        self.input_tokens += input or 0
        self.output_tokens += output or 0
        self.total_tokens += (total if total is not None else (input or 0) + (output or 0))
        self.cost_usd += cost or 0.0


@dataclass
class Step:
    """One tool invocation in the agent loop (observability)."""

    tool: str
    args: dict = field(default_factory=dict)
    result: str = ""


@dataclass
class Event:
    """The one neutral streaming event. ``type`` is text | tool_use | tool_result | done."""

    type: EventType
    text: str = ""                 # text: the assistant chunk/message
    name: str = ""                 # tool_use/tool_result: the tool name
    args: Optional[dict] = None     # tool_use: the arguments
    data: Any = None                # done: the final Result; else free-form


@dataclass
class Result:
    """What a run returns — a neutral reduction of the backend's message stream.
    ``str(result)`` is the final text, so ``print(result)`` works."""

    text: str = ""
    data: Any = None                            # structured payload when schema= was given
    files: dict[str, str] = field(default_factory=dict)
    status: Status = "done"
    error: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    steps: list[Step] = field(default_factory=list)
    session_id: Optional[str] = None

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        head = self.text[:60].replace("\n", " ")
        return (
            f"Result(status={self.status!r}, text={head!r}"
            f"{'…' if len(self.text) > 60 else ''}, files={list(self.files)}, "
            f"tokens={self.usage.total_tokens})"
        )
