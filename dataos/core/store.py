from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataos.core.artifact import Artifact


@dataclass(slots=True)
class LocalArtifactStore:
    workspace_dir: Path | str

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir)

    def prepare(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: Artifact) -> Artifact:
        artifact.save_meta()
        return artifact
