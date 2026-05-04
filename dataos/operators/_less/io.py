from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact

REQUIRED_COLUMNS = ("instruction", "response")


def find_target_dataset(inputs: list[Artifact], target_artifact_id: str) -> DatasetArtifact:
    for artifact in inputs:
        if isinstance(artifact, DatasetArtifact) and artifact.artifact_id == target_artifact_id:
            return artifact
    raise ValueError(f"less: target_artifact_id '{target_artifact_id}' not found in inputs")


def find_pool_dataset(inputs: list[Artifact], target_artifact_id: str) -> DatasetArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact) and artifact.artifact_id != target_artifact_id:
            return artifact
    raise ValueError(
        "less: requires a candidate pool DatasetArtifact distinct from the target in inputs"
    )


def select_train_split(dataset_obj: Dataset | DatasetDict) -> tuple[Dataset, str | None]:
    if isinstance(dataset_obj, DatasetDict):
        split = "train" if "train" in dataset_obj else next(iter(dataset_obj.keys()))
        return dataset_obj[split], split
    return dataset_obj, None


def check_required_columns(dataset_obj: Dataset | DatasetDict, role: str) -> None:
    flat, _ = select_train_split(dataset_obj)
    missing = [col for col in REQUIRED_COLUMNS if col not in flat.column_names]
    if missing:
        raise ValueError(
            f"less: {role} dataset is missing required columns: "
            f"{', '.join(missing)} (have: {', '.join(flat.column_names)})"
        )


def build_selected_dataset(
    dataset_obj: Dataset | DatasetDict,
    pool_split: str | None,
    indices: list[int],
) -> Dataset | DatasetDict:
    if isinstance(dataset_obj, DatasetDict):
        target_split = pool_split or "train"
        new_splits = {}
        for split_name, split_dataset in dataset_obj.items():
            if split_name == target_split:
                new_splits[split_name] = split_dataset.select(indices)
            else:
                new_splits[split_name] = split_dataset
        return DatasetDict(new_splits)
    return dataset_obj.select(indices)
