from __future__ import annotations

import json

from dataos.core.artifact import Artifact, DatasetArtifact, ModelArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox


def _find_latest_dataset(inputs: list[Artifact]) -> DatasetArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError("train_llamafactory requires at least one DatasetArtifact input")


class TrainLlamaFactoryOperator(Operator):
    name = "train_llamafactory"
    description = "Stub train a model from the latest dataset artifact."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> ModelArtifact:
        task = kwargs.get("task")
        dataset_artifact = _find_latest_dataset(inputs)
        dataset_summary = dataset_artifact.summary()
        dataset_obj = dataset_artifact.load()

        if dataset_summary["hf_object_type"] == "DatasetDict":
            train_split = "train" if "train" in dataset_obj else next(iter(dataset_obj.keys()))
            selected_rows = dataset_obj[train_split].num_rows
        else:
            train_split = "train"
            selected_rows = dataset_obj.num_rows

        artifact_id = make_artifact_id("model")
        model_dir = sandbox.artifact_dir("model", artifact_id)
        config = {
            "mode": kwargs.get("mode", "stub"),
            "dataset_artifact_id": dataset_artifact.artifact_id,
            "train_split": train_split,
            "selected_rows": selected_rows,
            "task_id": getattr(task, "task_id", None),
        }
        (model_dir / "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        (model_dir / "weights.stub").write_text("stub model binary placeholder\n", encoding="utf-8")

        model_artifact = ModelArtifact(
            artifact_id=artifact_id,
            name=f"model-{dataset_artifact.artifact_id}",
            path=model_dir,
            parent_ids=[dataset_artifact.artifact_id],
            producer=self.name,
            meta={
                "mode": config["mode"],
                "source_dataset_id": dataset_artifact.artifact_id,
                "train_split": train_split,
                "selected_rows": selected_rows,
                "task_id": getattr(task, "task_id", None),
            },
        )
        sandbox.log(
            "train_llamafactory",
            f"dataset={dataset_artifact.artifact_id}\nmodel={model_artifact.artifact_id}\nmode={config['mode']}\n",
        )
        return model_artifact
