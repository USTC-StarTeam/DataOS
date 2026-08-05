"""State persistence helpers for runtime-aware operators."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from dataos_operator.runtime.types import ArtifactRef, OperatorState


def snapshot_payload(state: OperatorState) -> dict[str, Any]:
    """Convert an operator state into a serializable payload."""

    return {
        "name": state.name,
        "data": dict(state.data),
        "metadata": dict(state.metadata),
        "artifacts": [
            {
                "name": artifact.name,
                "kind": artifact.kind,
                "path": artifact.path,
                "uri": artifact.uri,
                "metadata": dict(artifact.metadata),
            }
            for artifact in state.artifacts
        ],
    }


def restore_state(payload: dict[str, Any]) -> OperatorState:
    """Restore an operator state from a serialized payload."""

    artifacts = [
        ArtifactRef(
            name=item["name"],
            kind=item["kind"],
            path=item.get("path"),
            uri=item.get("uri"),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload.get("artifacts", [])
    ]
    return OperatorState(
        name=payload["name"],
        data=dict(payload.get("data", {})),
        metadata=dict(payload.get("metadata", {})),
        artifacts=artifacts,
    )


def save_state_snapshot(path: str | Path, states: dict[str, OperatorState]) -> Path:
    """Persist all operator states to disk with torch serialization."""

    torch = _torch_module()

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: snapshot_payload(state) for name, state in states.items()}, resolved)
    return resolved


def load_state_snapshot(path: str | Path) -> dict[str, OperatorState]:
    """Load all operator states from a snapshot file."""

    torch = _torch_module()

    payload = torch.load(Path(path), map_location="cpu")
    return {name: restore_state(state_payload) for name, state_payload in payload.items()}


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Torch is required for DataOS runtime state snapshots. "
            "Install dataos-operator with the `train` extra."
        ) from exc
