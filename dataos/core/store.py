from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dataos.core.artifact import Artifact, DatasetArtifact


ARTIFACT_DIRS = {
    "dataset": "datasets",
    "model": "models",
    "eval": "evals",
    "result": "results",
    "experiment": "experiments",
}


@dataclass(slots=True)
class LocalArtifactStore:
    workspace_dir: Path | str
    artifacts_root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir)
        self.artifacts_root = self.workspace_dir / "artifacts"

    def prepare(self) -> None:
        for dirname in ARTIFACT_DIRS.values():
            (self.artifacts_root / dirname).mkdir(parents=True, exist_ok=True)

    def alloc_path(self, artifact_type: str, artifact_id: str) -> Path:
        if artifact_type not in ARTIFACT_DIRS:
            raise ValueError(f"Unsupported artifact type for store: {artifact_type}")
        target = self.artifacts_root / ARTIFACT_DIRS[artifact_type] / artifact_id
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save(self, artifact: Artifact) -> Artifact:
        target = self.alloc_path(artifact.type, artifact.artifact_id)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(artifact.path, target)
        cloned = type(artifact)(
            artifact_id=artifact.artifact_id,
            name=artifact.name,
            path=target,
            meta=dict(artifact.meta),
            created_at=artifact.created_at,
            parent_ids=list(artifact.parent_ids),
            producer=artifact.producer,
        )
        if isinstance(cloned, DatasetArtifact):
            cloned.meta.setdefault("format", "hf_datasets")
        cloned.save_meta()
        return cloned
