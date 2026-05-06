from __future__ import annotations

import random

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import ensure_dataset_dict, find_latest_dataset, restore_dataset_type


def _interleave_by_category(groups: dict[str, list[dict]]) -> list[dict]:
    categories = [category for category, rows in groups.items() if rows]
    indices = {category: 0 for category in categories}
    output: list[dict] = []
    while categories:
        next_categories: list[str] = []
        for category in categories:
            index = indices[category]
            rows = groups[category]
            if index < len(rows):
                output.append(rows[index])
                indices[category] = index + 1
            if indices[category] < len(rows):
                next_categories.append(category)
        categories = next_categories
    return output


class CategoryBalanceOperator(Operator):
    name = "category_balance"
    description = "Rebalance a split by category using deterministic downsample or upsample strategies."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        split_name = str(kwargs.get("split", "train"))
        category_column = str(kwargs.get("category_column", "dataos_category"))
        strategy = str(kwargs.get("strategy", "downsample_to_min"))
        if strategy not in {"downsample_to_min", "cap", "upsample_to_max"}:
            raise ValueError("category_balance: unsupported strategy")
        seed = int(kwargs.get("seed", 42))
        target_per_category = kwargs.get("target_per_category")

        transformed = dict(dataset_dict)
        if split_name not in dataset_dict:
            raise ValueError(f"category_balance: split '{split_name}' not found")
        split_dataset = dataset_dict[split_name]
        if category_column not in split_dataset.column_names:
            raise ValueError(f"category_balance: category_column '{category_column}' not found")

        rng = random.Random(seed)
        groups: dict[str, list[dict]] = {}
        for row in split_dataset:
            category = str(row.get(category_column, "unknown"))
            groups.setdefault(category, []).append(dict(row))
        for rows in groups.values():
            rng.shuffle(rows)

        counts = {category: len(rows) for category, rows in groups.items()}
        if strategy == "downsample_to_min":
            target = min(counts.values()) if counts else 0
        elif target_per_category is None:
            raise ValueError("category_balance: target_per_category is required for this strategy")
        else:
            target = int(target_per_category)

        balanced: dict[str, list[dict]] = {}
        for category, rows in groups.items():
            if strategy in {"downsample_to_min", "cap"}:
                balanced[category] = rows[: min(len(rows), target)]
            else:
                expanded = list(rows)
                while rows and len(expanded) < target:
                    expanded.append(dict(rows[(len(expanded) - len(rows)) % len(rows)]))
                balanced[category] = expanded[:target]

        balanced_rows = _interleave_by_category(balanced)
        transformed[split_name] = Dataset.from_list(balanced_rows)

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-category-balance",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "category_balance_split": split_name,
                "category_balance_category_column": category_column,
                "category_balance_strategy": strategy,
                "category_balance_seed": seed,
                "category_balance_input_counts": counts,
                "category_balance_output_counts": {category: len(rows) for category, rows in balanced.items()},
            },
        )
        artifact.save(output_obj)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nstrategy={strategy}\ninput_counts={counts}\noutput_counts={artifact.meta['category_balance_output_counts']}\n",
        )
        return artifact
