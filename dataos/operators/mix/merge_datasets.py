from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id, select_working_dataset
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._dataset_utils import copy_row, detect_dataset_format, find_dataset_artifacts


class MergeDatasetsOperator(Operator):
    name = "merge_datasets"
    category = "mix"
    description = "Merge all dataset artifacts into a single working dataset with source tracking metadata."
    chinese_description = "将多个数据集工件合并成一个单一工作数据集，并保留来源信息。"
    parameter_specs = []

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        del kwargs
        sources = find_dataset_artifacts(inputs, self.name)
        merged_rows: list[dict] = []
        source_summaries: list[dict[str, object]] = []

        for source in sources:
            dataset_obj = source.load()
            if isinstance(dataset_obj, DatasetDict):
                dataset, selected_split, available_splits = select_working_dataset(dataset_obj)
            elif isinstance(dataset_obj, Dataset):
                dataset = dataset_obj
                selected_split = str(source.meta.get("source_selected_split") or "train")
                available_splits = list(source.meta.get("source_available_splits") or [selected_split])
            else:
                raise ValueError(
                    f"merge_datasets expects Dataset inputs, got {type(dataset_obj)!r} from {source.artifact_id}"
                )

            row_count = dataset.num_rows
            dataset_format = detect_dataset_format(list(dataset.column_names), source.meta)
            if dataset_format != "sharegpt":
                raise ValueError(
                    f"merge_datasets expects sharegpt inputs, got {dataset_format!r} from {source.artifact_id}"
                )

            source_name = str(source.meta.get("source") or source.name)
            source_summaries.append(
                {
                    "artifact_id": source.artifact_id,
                    "name": source.name,
                    "source": source_name,
                    "selected_split": selected_split,
                    "available_splits": available_splits,
                    "rows": row_count,
                }
            )

            for index, row in enumerate(dataset):
                original_row = copy_row(dict(row))
                conversations = original_row.get("conversations")
                if not isinstance(conversations, list):
                    raise ValueError(
                        f"merge_datasets: row {index} in {source.artifact_id} is missing a conversations list"
                    )
                merged_row = {
                    "conversations": copy_row(conversations),
                    "source_dataset_name": source_name,
                    "source_artifact_id": source.artifact_id,
                    "source_row_idx": index,
                }
                if isinstance(original_row.get("tags"), list):
                    merged_row["tags"] = copy_row(original_row["tags"])
                merged_rows.append(merged_row)

        merged_dataset = Dataset.from_list(merged_rows)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name="merged-working-dataset",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id for source in sources],
            producer=self.name,
            meta={
                "dataset_format": "sharegpt",
                "mix_name": self.name,
                "mix_sources": source_summaries,
                "mix_total_rows": len(merged_rows),
            },
        )
        artifact.save(merged_dataset)
        sandbox.log(
            self.name,
            "\n".join(
                [
                    f"sources={[source.artifact_id for source in sources]}",
                    f"output={artifact.artifact_id}",
                    f"rows={len(merged_rows)}",
                ]
            ) + "\n",
        )
        return artifact
