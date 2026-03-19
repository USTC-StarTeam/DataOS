"""Lineage storage interfaces shared by runtime and controller."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dataos.core.dataset import DatasetVersion, LineageRecord

__all__ = ["LineageStore"]


@runtime_checkable
class LineageStore(Protocol):
    """Boundary contract for persisting and querying dataset lineage."""

    def save_lineage(self, dataset: DatasetVersion) -> None:
        """Persist lineage for one dataset version."""

    def load_lineage(self, dataset_id: str, version: str) -> list[LineageRecord]:
        """Load lineage records for one dataset version."""
