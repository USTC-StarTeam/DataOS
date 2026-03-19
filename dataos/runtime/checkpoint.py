"""Checkpoint interfaces shared by runtime recovery components."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dataos.storage.run_store import RunSnapshot

__all__ = [
    "CheckpointManager",
    "CheckpointRecord",
]


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(value)


@dataclass(slots=True)
class CheckpointRecord:
    """Structured reference to a persisted runtime checkpoint."""

    checkpoint_id: str
    run_id: str
    step_id: str | None = None
    payload_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        self.metadata = _copy_mapping(self.metadata)


@runtime_checkable
class CheckpointManager(Protocol):
    """Boundary contract for creating and restoring runtime checkpoints."""

    def create(self, snapshot: RunSnapshot) -> CheckpointRecord:
        """Persist one checkpoint for the given runtime snapshot."""

    def restore(self, checkpoint_id: str) -> RunSnapshot:
        """Restore the runtime snapshot identified by checkpoint ID."""
