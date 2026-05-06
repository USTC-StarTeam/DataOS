from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, make_artifact_id


def build_source(tmp_path, dataset_obj, name: str = "src") -> DatasetArtifact:
    artifact_id = make_artifact_id("dataset")
    path = tmp_path / "src" / artifact_id
    path.mkdir(parents=True)
    artifact = DatasetArtifact(
        artifact_id=artifact_id,
        name=name,
        path=path,
        parent_ids=[],
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact
