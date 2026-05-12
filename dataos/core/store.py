from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dataos.core.artifact import Artifact


@dataclass(slots=True)
class LocalArtifactStore:
    workspace_dir: Path | str
    experiments_root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir)
        self.experiments_root = self.workspace_dir / "experiments"

    def prepare(self) -> None:
        self.experiments_root.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: Artifact) -> Artifact:
        artifact.save_meta()
        return artifact
