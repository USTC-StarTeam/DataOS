from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact
from dataos.core.operator import Operator, OperatorParameter
from dataos.core.sandbox import Sandbox
from dataos.operators._dataset_utils import (
    ensure_dataset_dict,
    find_latest_dataset,
    normalize_text,
    restore_dataset_type,
    save_dataset_artifact,
)


class FilterExactDedupOperator(Operator):
    name = "filter_exact_dedup"
    category = "filter"
    description = "Remove exact duplicate rows using a composite key built from selected columns."
    chinese_description = "按指定字段组合成 key，对完全重复的样本做精确去重。"
    parameter_specs = [
        OperatorParameter(
            name="keys",
            type="list[str]",
            description="Columns used to build the deduplication key.",
            required=True,
        )
    ]

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        keys = [str(key) for key in kwargs["keys"]]
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)

        transformed: dict[str, Dataset] = {}
        removed_counts: dict[str, int] = {}
        for split, dataset in dataset_dict.items():
            for key in keys:
                if key not in dataset.column_names:
                    raise ValueError(f"filter_exact_dedup: column '{key}' not found in split '{split}'")
            seen: set[str] = set()
            rows: list[dict] = []
            removed = 0
            for row in dataset:
                dedup_key = "\x1f".join(normalize_text(row.get(key)).lower() for key in keys)
                if dedup_key in seen:
                    removed += 1
                    continue
                seen.add(dedup_key)
                rows.append(dict(row))
            transformed[split] = Dataset.from_list(rows)
            removed_counts[split] = removed

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact = save_dataset_artifact(
            source=source,
            sandbox=sandbox,
            operator_name=self.name,
            suffix="dedup",
            dataset_obj=output_obj,
            meta={
                "filter_name": self.name,
                "filter_keys": keys,
                "filter_removed_counts": removed_counts,
                "dataset_format": source.meta.get("dataset_format", "alpaca"),
            },
        )
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nremoved={removed_counts}\n",
        )
        return artifact
