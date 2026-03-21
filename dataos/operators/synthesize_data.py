from __future__ import annotations

from datasets import Dataset, DatasetDict, concatenate_datasets

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox


def _find_first_dataset(inputs: list[Artifact]) -> DatasetArtifact:
    for artifact in inputs:
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError("synthesize_data requires at least one DatasetArtifact input")


def _augment_split(dataset: Dataset, ratio: float) -> tuple[Dataset, int]:
    if dataset.num_rows == 0 or ratio <= 0:
        return dataset, 0
    extra_count = max(1, int(dataset.num_rows * ratio))
    indices = [index % dataset.num_rows for index in range(extra_count)]
    synthetic = dataset.select(indices)
    return concatenate_datasets([dataset, synthetic]), extra_count


class SynthesizeDataOperator(Operator):
    name = "synthesize_data"
    description = "Apply simple rule-based augmentation to a Hugging Face dataset artifact."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = _find_first_dataset(inputs)
        ratio = float(kwargs.get("augment_ratio", 0.25))
        dataset_obj = source.load()

        augmentation_counts: dict[str, int] = {}
        if isinstance(dataset_obj, DatasetDict):
            target_split = "train" if "train" in dataset_obj else next(iter(dataset_obj.keys()))
            augmented_splits: dict[str, Dataset] = {}
            for split, split_dataset in dataset_obj.items():
                if split == target_split:
                    augmented_splits[split], augmentation_counts[split] = _augment_split(split_dataset, ratio)
                else:
                    augmented_splits[split] = split_dataset
                    augmentation_counts[split] = 0
            augmented = DatasetDict(augmented_splits)
        else:
            augmented_dataset, extra_count = _augment_split(dataset_obj, ratio)
            augmented = augmented_dataset
            augmentation_counts = {"train": extra_count}

        artifact_id = make_artifact_id("dataset")
        target = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-synthetic",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "augmentation_ratio": ratio,
                "augmentation_counts": augmentation_counts,
                "augmentation_source_artifact_id": source.artifact_id,
            },
        )
        target.save(augmented)
        sandbox.log(
            "synthesize_data",
            f"source={source.artifact_id}\noutput={target.artifact_id}\nratio={ratio}\ncounts={augmentation_counts}\n",
        )
        return target
