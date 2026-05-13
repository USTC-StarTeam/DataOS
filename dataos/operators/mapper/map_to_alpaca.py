from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact
from dataos.core.operator import Operator, OperatorParameter
from dataos.core.sandbox import Sandbox
from dataos.operators._dataset_utils import (
    build_alpaca_row,
    ensure_dataset_dict,
    extract_text_pair,
    find_latest_dataset,
    preserve_context_fields,
    restore_dataset_type,
    save_dataset_artifact,
)


class MapToAlpacaOperator(Operator):
    name = "map_to_alpaca"
    category = "mapper"
    description = "Convert each row into an Alpaca-style instruction/input/output schema."
    chinese_description = "将数据集统一映射成 Alpaca 风格的 instruction、input、output 列。"
    parameter_specs = [
        OperatorParameter(
            name="keep_other_columns",
            type="bool",
            description="Whether to preserve non-core source columns on each output row.",
            default=False,
        )
    ]

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        keep_other_columns = bool(kwargs.get("keep_other_columns", False))
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)

        transformed: dict[str, Dataset] = {}
        row_counts: dict[str, int] = {}
        for split, dataset in dataset_dict.items():
            rows: list[dict] = []
            for row in dataset:
                source_row = dict(row)
                prompt, answer, system = extract_text_pair(source_row, list(dataset.column_names), source.meta)
                new_row = build_alpaca_row(prompt, answer, system=system)
                preserve_context_fields(source_row, new_row, keep_other_columns=keep_other_columns)
                rows.append(new_row)
            transformed[split] = Dataset.from_list(rows)
            row_counts[split] = len(rows)

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact = save_dataset_artifact(
            source=source,
            sandbox=sandbox,
            operator_name=self.name,
            suffix="alpaca",
            dataset_obj=output_obj,
            meta={
                "dataset_format": "alpaca",
                "mapper_name": self.name,
                "mapper_keep_other_columns": keep_other_columns,
                "mapper_row_counts": row_counts,
            },
        )
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nrows={row_counts}\n",
        )
        return artifact
