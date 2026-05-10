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


class SelectTopByScoreOperator(Operator):
    name = "select_top_by_score"
    category = "selector"
    description = "Select the top-scoring rows from each split based on an existing numeric column."
    chinese_description = "按照已有分数字段，从每个 split 中选择得分最高的样本。"
    parameter_specs = [
        OperatorParameter(
            name="score_column",
            type="str",
            description="Numeric column used for ranking rows.",
            default="quality_score",
        ),
        OperatorParameter(
            name="max_rows",
            type="int",
            description="Maximum number of rows to keep per split.",
            required=True,
        ),
        OperatorParameter(
            name="descending",
            type="bool",
            description="Whether larger scores are better.",
            default=True,
        ),
    ]

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        score_column = str(kwargs.get("score_column", "quality_score"))
        max_rows = int(kwargs["max_rows"])
        descending = bool(kwargs.get("descending", True))
        if max_rows < 0:
            raise ValueError("select_top_by_score: max_rows must be >= 0")

        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        transformed: dict[str, Dataset] = {}
        kept_counts: dict[str, int] = {}

        for split, dataset in dataset_dict.items():
            if score_column not in dataset.column_names:
                raise ValueError(f"select_top_by_score: column '{score_column}' not found in split '{split}'")
            rows = sorted(dataset, key=lambda row: float(row.get(score_column) or 0.0), reverse=descending)
            transformed[split] = Dataset.from_list(rows[:max_rows])
            kept_counts[split] = min(len(rows), max_rows)

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact = save_dataset_artifact(
            source=source,
            sandbox=sandbox,
            operator_name=self.name,
            suffix="top-by-score",
            dataset_obj=output_obj,
            meta={
                "selector_name": self.name,
                "selector_score_column": score_column,
                "selector_max_rows": max_rows,
                "selector_descending": descending,
                "selector_kept_counts": kept_counts,
                "dataset_format": source.meta.get("dataset_format", "alpaca"),
            },
        )
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nscore_column={score_column}\nkept={kept_counts}\n",
        )
        return artifact
