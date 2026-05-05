from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import detect_language_label, ensure_dataset_dict, find_latest_dataset, infer_text_columns, join_text_columns, restore_dataset_type


class LangFilterOperator(Operator):
    name = "lang_filter"
    description = "Keep rows whose dominant script matches the requested language label, with optional annotation."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        allowed_languages = list(kwargs.get("allowed_languages") or ["en"])
        min_fraction = float(kwargs.get("min_fraction", 0.6))
        annotate_column = kwargs.get("annotate_column")

        transformed: dict[str, Dataset] = {}
        kept_counts: dict[str, int] = {}
        removed_counts: dict[str, int] = {}
        detected_counts: dict[str, dict[str, int]] = {}
        column_meta: dict[str, list[str]] = {}

        for split, dataset in dataset_dict.items():
            columns = list(kwargs.get("columns") or infer_text_columns(dataset, ["instruction", "input", "response", "output"]))
            column_meta[split] = columns
            rows: list[dict] = []
            counts = {"en": 0, "zh": 0, "mixed": 0, "other": 0}
            for row in dataset:
                label = detect_language_label(join_text_columns(row, columns), min_fraction=min_fraction)
                counts[label] = counts.get(label, 0) + 1
                if label not in allowed_languages:
                    continue
                new_row = dict(row)
                if annotate_column:
                    new_row[str(annotate_column)] = label
                rows.append(new_row)
            transformed[split] = Dataset.from_list(rows)
            kept_counts[split] = len(rows)
            removed_counts[split] = dataset.num_rows - len(rows)
            detected_counts[split] = counts

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-lang-filter",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "lang_filter_allowed_languages": allowed_languages,
                "lang_filter_min_fraction": min_fraction,
                "lang_filter_columns": column_meta,
                "lang_filter_detected_counts": detected_counts,
                "lang_filter_kept_counts": kept_counts,
                "lang_filter_removed_counts": removed_counts,
                "lang_filter_annotate_column": annotate_column,
            },
        )
        artifact.save(output_obj)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nallowed={allowed_languages}\nkept={kept_counts}\nremoved={removed_counts}\n",
        )
        return artifact
