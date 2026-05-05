from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import count_words, ensure_dataset_dict, find_latest_dataset, join_text_columns, restore_dataset_type


def _surface_complexity(row: dict) -> float:
    prompt = join_text_columns(row, ["instruction", "input"])
    response = str(row.get("response") or row.get("output") or "")
    score = 0.0
    score += count_words(prompt) * 1.0
    score += count_words(response) * 0.5
    score += prompt.count("\n") * 2.0
    score += response.count("\n") * 1.0
    score += 5.0 if "```" in prompt or "```" in response else 0.0
    return round(score, 4)


class CurriculumSortOperator(Operator):
    name = "curriculum_sort"
    description = "Order a split from easy-to-hard or hard-to-easy using a simple surface-complexity proxy."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        split_name = str(kwargs.get("split", "train"))
        order = str(kwargs.get("order", "easy_to_hard"))
        if order not in {"easy_to_hard", "hard_to_easy"}:
            raise ValueError("curriculum_sort: order must be 'easy_to_hard' or 'hard_to_easy'")
        score_column = str(kwargs.get("score_column", "dataos_curriculum_score"))
        if split_name not in dataset_dict:
            raise ValueError(f"curriculum_sort: split '{split_name}' not found")

        transformed = dict(dataset_dict)
        rows: list[dict] = []
        for row in dataset_dict[split_name]:
            new_row = dict(row)
            new_row[score_column] = _surface_complexity(new_row)
            rows.append(new_row)
        rows.sort(key=lambda row: row[score_column], reverse=(order == "hard_to_easy"))
        transformed[split_name] = Dataset.from_list(rows)

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-curriculum-sort",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "curriculum_sort_split": split_name,
                "curriculum_sort_order": order,
                "curriculum_sort_score_column": score_column,
                "curriculum_sort_min_score": min((row[score_column] for row in rows), default=0.0),
                "curriculum_sort_max_score": max((row[score_column] for row in rows), default=0.0),
            },
        )
        artifact.save(output_obj)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nsplit={split_name}\norder={order}\nrows={len(rows)}\n",
        )
        return artifact
