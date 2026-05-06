from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import count_chars, count_words, ensure_dataset_dict, find_latest_dataset, infer_text_columns, join_text_columns, restore_dataset_type


class LengthFilterOperator(Operator):
    name = "length_filter"
    description = "Filter rows by token or character length across selected text columns."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        unit = str(kwargs.get("unit", "words"))
        if unit not in {"words", "chars"}:
            raise ValueError("length_filter: unit must be 'words' or 'chars'")
        min_length = kwargs.get("min_length")
        max_length = kwargs.get("max_length")
        per_column_min = dict(kwargs.get("per_column_min") or {})
        per_column_max = dict(kwargs.get("per_column_max") or {})
        metric = count_words if unit == "words" else count_chars

        transformed: dict[str, Dataset] = {}
        kept_counts: dict[str, int] = {}
        removed_counts: dict[str, int] = {}
        column_meta: dict[str, list[str]] = {}

        for split, dataset in dataset_dict.items():
            columns = list(kwargs.get("columns") or infer_text_columns(dataset, ["instruction", "input", "response", "output"]))
            column_meta[split] = columns
            keep_indices: list[int] = []
            for index, row in enumerate(dataset):
                total = metric(join_text_columns(row, columns))
                if min_length is not None and total < int(min_length):
                    continue
                if max_length is not None and total > int(max_length):
                    continue
                failed = False
                for column, limit in per_column_min.items():
                    if metric(str(row.get(column, ""))) < int(limit):
                        failed = True
                        break
                if failed:
                    continue
                for column, limit in per_column_max.items():
                    if metric(str(row.get(column, ""))) > int(limit):
                        failed = True
                        break
                if failed:
                    continue
                keep_indices.append(index)
            transformed[split] = dataset.select(keep_indices)
            kept_counts[split] = len(keep_indices)
            removed_counts[split] = dataset.num_rows - len(keep_indices)

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-length-filter",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "length_filter_unit": unit,
                "length_filter_columns": column_meta,
                "length_filter_min_length": min_length,
                "length_filter_max_length": max_length,
                "length_filter_per_column_min": per_column_min,
                "length_filter_per_column_max": per_column_max,
                "length_filter_kept_counts": kept_counts,
                "length_filter_removed_counts": removed_counts,
            },
        )
        artifact.save(output_obj)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nunit={unit}\nkept={kept_counts}\nremoved={removed_counts}\n",
        )
        return artifact
