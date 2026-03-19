"""Runtime state persistence interfaces for checkpoints and resume."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dataos.storage.run_store import RunSnapshot

__all__ = ["RuntimeStateStore"]


@runtime_checkable
class RuntimeStateStore(Protocol):
    """Boundary contract for persisting and loading in-flight runtime state."""

    def save(self, snapshot: RunSnapshot) -> None:
        """Persist the latest runtime snapshot for a run."""

    def load(self, run_id: str) -> RunSnapshot | None:
        """Load the latest runtime snapshot for a run if it exists."""
