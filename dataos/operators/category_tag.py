from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import ensure_dataset_dict, find_latest_dataset, restore_dataset_type

_PRESET_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("translation", ("translate", "translation", "翻译")),
    ("math", ("solve", "calculate", "equation", "math", "+", "-", "*", "/")),
    ("code", ("python", "code", "function", "bug", "algorithm", "sql", "program")),
    ("classification", ("classify", "label", "sentiment", "category")),
    ("extraction", ("extract", "identify", "find all", "from the passage")),
    ("writing", ("write", "draft", "compose", "summarize", "essay")),
]


def _label_text(text: str, default_category: str) -> str:
    lowered = text.lower()
    for category, patterns in _PRESET_RULES:
        if any(pattern in lowered for pattern in patterns):
            return category
    return default_category


class CategoryTagOperator(Operator):
    name = "category_tag"
    description = "Annotate rows with a lightweight task-surface category for downstream balancing and analysis."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        source_column = str(kwargs.get("source_column", "instruction"))
        target_column = str(kwargs.get("target_column", "dataos_category"))
        default_category = str(kwargs.get("default_category", "qa"))

        transformed: dict[str, Dataset] = {}
        counts_by_split: dict[str, dict[str, int]] = {}
        for split, dataset in dataset_dict.items():
            if source_column not in dataset.column_names:
                raise ValueError(f"category_tag: source_column '{source_column}' not found in split '{split}'")
            rows: list[dict] = []
            counts: dict[str, int] = {}
            for row in dataset:
                label = _label_text(str(row.get(source_column, "")), default_category)
                counts[label] = counts.get(label, 0) + 1
                new_row = dict(row)
                new_row[target_column] = label
                rows.append(new_row)
            transformed[split] = Dataset.from_list(rows)
            counts_by_split[split] = counts

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-category-tag",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "category_tag_source_column": source_column,
                "category_tag_target_column": target_column,
                "category_tag_default": default_category,
                "category_tag_counts": counts_by_split,
            },
        )
        artifact.save(output_obj)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\ncounts={counts_by_split}\n",
        )
        return artifact
