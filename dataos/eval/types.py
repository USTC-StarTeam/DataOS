from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelLike(Protocol):
    """Structural type for a trained model the eval subsystem consumes.

    The eval backend only reads these four attributes, so any object exposing
    them is accepted. Importing this Protocol instead of a concrete artifact
    class also breaks the eval -> core.artifact import cycle without touching
    any function body.
    """

    path: Path
    meta: dict[str, Any]
    artifact_id: str
    producer: str


@dataclass(slots=True)
class LmEvalRunOutput:
    """One lm-eval run, normalized: per-task results + samples + the raw payload
    + a protocol snapshot (model args / seeds / skipped tasks) for the receipts."""

    tasks: list[str]
    results: dict[str, Any]
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
