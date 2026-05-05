from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import collect_dataset_artifacts, ensure_dataset_dict, round_robin_merge, restore_dataset_type


def _select_sources(inputs: list[Artifact], source_artifact_ids: list[str] | None) -> list[DatasetArtifact]:
    datasets = collect_dataset_artifacts(inputs)
    if not datasets:
        raise ValueError("mixture requires at least one DatasetArtifact input")
    if not source_artifact_ids:
        return datasets
    selected: list[DatasetArtifact] = []
    by_id = {artifact.artifact_id: artifact for artifact in datasets}
    for artifact_id in source_artifact_ids:
        if artifact_id not in by_id:
            raise ValueError(f"mixture: source artifact '{artifact_id}' not found in inputs")
        selected.append(by_id[artifact_id])
    return selected


class MixtureOperator(Operator):
    name = "mixture"
    description = "Create an artifact-aware training mixture from multiple upstream datasets with source provenance."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source_ids = kwargs.get("source_artifact_ids")
        if source_ids is not None:
            source_ids = list(source_ids)
        sources = _select_sources(inputs, source_ids)
        split_name = str(kwargs.get("split", "train"))
        max_rows = kwargs.get("max_rows")
        if max_rows is None:
            raise ValueError("mixture: max_rows is required")
        max_rows = int(max_rows)
        source_column = str(kwargs.get("source_column", "dataos_source_artifact"))
        preserve_other_splits = bool(kwargs.get("preserve_other_splits", True))

        records_by_source: dict[str, list[dict]] = {}
        other_splits: dict[str, Dataset] = {}
        weights = dict(kwargs.get("weights") or {})
        for artifact in sources:
            dataset_obj = artifact.load()
            dataset_dict, _ = ensure_dataset_dict(dataset_obj)
            if split_name not in dataset_dict:
                raise ValueError(f"mixture: split '{split_name}' not found in artifact '{artifact.artifact_id}'")
            records: list[dict] = []
            for row in dataset_dict[split_name]:
                new_row = dict(row)
                new_row[source_column] = artifact.artifact_id
                records.append(new_row)
            multiplier = max(1, int(weights.get(artifact.artifact_id, 1)))
            records_by_source[artifact.artifact_id] = records * multiplier
            if preserve_other_splits:
                for split, split_dataset in dataset_dict.items():
                    if split != split_name and split not in other_splits:
                        other_splits[split] = split_dataset

        mixed_rows = round_robin_merge(records_by_source, max_rows)
        mixed_dataset = Dataset.from_list(mixed_rows)
        output_dict = DatasetDict({split_name: mixed_dataset, **other_splits})
        output_obj = restore_dataset_type(output_dict, True)

        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name="mixture",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[artifact.artifact_id for artifact in sources],
            producer=self.name,
            meta={
                "mixture_split": split_name,
                "mixture_max_rows": max_rows,
                "mixture_source_column": source_column,
                "mixture_source_ids": [artifact.artifact_id for artifact in sources],
                "mixture_weights": {artifact.artifact_id: int(weights.get(artifact.artifact_id, 1)) for artifact in sources},
                "mixture_output_rows": len(mixed_rows),
            },
        )
        artifact.save(output_obj)
        sandbox.log(
            self.name,
            f"sources={[artifact.artifact_id for artifact in sources]}\noutput={artifact.artifact_id}\nsplit={split_name}\nrows={len(mixed_rows)}\n",
        )
        return artifact
