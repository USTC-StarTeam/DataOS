"""Artifact storage interfaces shared by runtime, eval, and controller."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["ArtifactStore"]


@runtime_checkable
class ArtifactStore(Protocol):
    """Boundary contract for persisting opaque artifacts and returning their URIs."""

    def save_artifact(self, namespace: str, name: str, payload: bytes | str) -> str:
        """Persist an artifact payload and return a stable URI."""

    def load_artifact(self, uri: str) -> bytes | str:
        """Load a previously stored artifact payload."""
