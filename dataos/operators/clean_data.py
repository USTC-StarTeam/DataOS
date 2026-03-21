from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox


def _find_first_dataset(inputs: list[Artifact]) -> DatasetArtifact:
    for artifact in inputs:
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError("clean_data requires at least one DatasetArtifact input")


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in example.items():
        target_key = _normalize_column_name(key)
        if isinstance(value, str):
            normalized[target_key] = value.strip()
        else:
            normalized[target_key] = value
    return normalized


def _row_has_signal(example: dict[str, Any]) -> bool:
    for value in example.values():
        if isinstance(value, str) and value.strip():
            return True
        if value not in (None, "", [], {}):
            return True
    return False


def _clean_dataset(dataset: Dataset) -> Dataset:
    cleaned = dataset.map(_normalize_example)
    renamed_columns = {
        column_name: _normalize_column_name(column_name)
        for column_name in cleaned.column_names
        if _normalize_column_name(column_name) != column_name
    }
    if renamed_columns:
        cleaned = cleaned.rename_columns(renamed_columns)
    return cleaned.filter(_row_has_signal)


class CleanDataOperator(Operator):
    name = "clean_data"
    description = "Perform minimal cleaning on a Hugging Face dataset artifact."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        del kwargs
        source = _find_first_dataset(inputs)
        dataset_obj = source.load()
        if isinstance(dataset_obj, DatasetDict):
            cleaned = DatasetDict({split: _clean_dataset(split_dataset) for split, split_dataset in dataset_obj.items()})
        else:
            cleaned = _clean_dataset(dataset_obj)

        artifact_id = make_artifact_id("dataset")
        target = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-clean",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={"cleaning_applied": ["strip_strings", "normalize_column_names", "drop_empty_rows"]},
        )
        target.save(cleaned)
        sandbox.log(
            "clean_data",
            f"source={source.artifact_id}\noutput={target.artifact_id}\nrows={target.meta.get('num_rows')}\n",
        )
        return target
