from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact
from dataos.core.operator import Operator, OperatorParameter
from dataos.core.sandbox import Sandbox
from dataos.operators._dataset_utils import (
    ensure_dataset_dict,
    find_latest_dataset,
    restore_dataset_type,
    save_dataset_artifact,
)


class SelectRandomSubsetOperator(Operator):
    name = "select_random_subset"
    category = "selector"
    description = "Select a deterministic random subset from each split."
    chinese_description = "按固定随机种子从每个 split 中选择一个子集。"
    parameter_specs = [
        OperatorParameter(
            name="max_rows",
            type="int",
            description="Maximum number of rows to keep per split.",
            required=True,
        ),
        OperatorParameter(
            name="seed",
            type="int",
            description="Random seed used for deterministic shuffling.",
            default=42,
        ),
    ]

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        max_rows = int(kwargs["max_rows"])
        seed = int(kwargs.get("seed", 42))
        if max_rows < 0:
            raise ValueError("select_random_subset: max_rows must be >= 0")

        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        transformed: dict[str, Dataset] = {}
        kept_counts: dict[str, int] = {}

        for split, dataset in dataset_dict.items():
            sampled = dataset if dataset.num_rows <= max_rows else dataset.shuffle(seed=seed).select(range(max_rows))
            transformed[split] = sampled
            kept_counts[split] = sampled.num_rows

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact = save_dataset_artifact(
            source=source,
            sandbox=sandbox,
            operator_name=self.name,
            suffix="random-subset",
            dataset_obj=output_obj,
            meta={
                "selector_name": self.name,
                "selector_max_rows": max_rows,
                "selector_seed": seed,
                "selector_kept_counts": kept_counts,
                "dataset_format": source.meta.get("dataset_format", "alpaca"),
            },
        )
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nkept={kept_counts}\n",
        )
        return artifact
