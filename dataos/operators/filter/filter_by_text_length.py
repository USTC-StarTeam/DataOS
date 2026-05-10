from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact
from dataos.core.operator import Operator, OperatorParameter
from dataos.core.sandbox import Sandbox
from dataos.operators._dataset_utils import (
    count_words,
    ensure_dataset_dict,
    extract_text_pair,
    find_latest_dataset,
    restore_dataset_type,
    save_dataset_artifact,
)


class FilterByTextLengthOperator(Operator):
    name = "filter_by_text_length"
    category = "filter"
    description = "Filter rows whose combined prompt/answer length falls outside a configured range."
    chinese_description = "按照 prompt 和 answer 的合并长度过滤过短或过长样本。"
    parameter_specs = [
        OperatorParameter(
            name="min_words",
            type="int",
            description="Minimum allowed word count for prompt plus answer.",
            default=1,
        ),
        OperatorParameter(
            name="max_words",
            type="int",
            description="Maximum allowed word count for prompt plus answer.",
            default=2048,
        ),
    ]

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        min_words = int(kwargs.get("min_words", 1))
        max_words = int(kwargs.get("max_words", 2048))
        if min_words < 0 or max_words < min_words:
            raise ValueError("filter_by_text_length: invalid min_words/max_words range")

        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        transformed: dict[str, Dataset] = {}
        kept_counts: dict[str, int] = {}
        removed_counts: dict[str, int] = {}

        for split, dataset in dataset_dict.items():
            rows: list[dict] = []
            removed = 0
            for row in dataset:
                prompt, answer, _ = extract_text_pair(dict(row), list(dataset.column_names), source.meta)
                total_words = count_words(prompt) + count_words(answer)
                if total_words < min_words or total_words > max_words:
                    removed += 1
                    continue
                rows.append(dict(row))
            transformed[split] = Dataset.from_list(rows)
            kept_counts[split] = len(rows)
            removed_counts[split] = removed

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact = save_dataset_artifact(
            source=source,
            sandbox=sandbox,
            operator_name=self.name,
            suffix="length-filter",
            dataset_obj=output_obj,
            meta={
                "filter_name": self.name,
                "filter_min_words": min_words,
                "filter_max_words": max_words,
                "filter_kept_counts": kept_counts,
                "filter_removed_counts": removed_counts,
                "dataset_format": source.meta.get("dataset_format", "alpaca"),
            },
        )
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nkept={kept_counts}\nremoved={removed_counts}\n",
        )
        return artifact
