from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalAdapterConfig:
    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRequestConfig:
    type: str
    batch_size: int = 8
    max_new_tokens: int = 256
    do_sample: bool = False


@dataclass(slots=True)
class AdapterCallContext:
    """Per-call envelope passed to adapter ``generate_until`` /
    ``score_continuations``.

    Carries the adapter knobs and the request shape; intentionally distinct
    from :class:`dataos.eval.config.EvalConfig` (the lm-eval-shaped task
    config). The bridge fills this in from a live ``DataOSLM`` instance — no
    user-facing surface should construct it directly.
    """

    adapter: EvalAdapterConfig | None = None
    request: EvalRequestConfig | None = None


@dataclass(slots=True)
class EvalRequest:
    sample_id: str
    index: int
    request_type: str
    prompt: str
    stop: list[str] | None = None
    choices: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelResponse:
    sample_id: str
    index: int
    request_type: str
    text: str | None = None
    logprob: float | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
