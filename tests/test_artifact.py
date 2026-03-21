from __future__ import annotations

import json

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact


def test_dataset_artifact_save_load_and_summary_for_dataset(tmp_path):
    dataset = Dataset.from_dict({"instruction": ["a", "b"], "response": ["x", "y"]})
    artifact = DatasetArtifact(
        artifact_id="dataset-1",
        name="demo-dataset",
        path=tmp_path / "dataset_artifact",
        producer="test",
    )

    artifact.save(dataset)
    restored = artifact.load()
    summary = artifact.summary()

    assert restored.num_rows == 2
    assert summary["hf_object_type"] == "Dataset"
    assert summary["splits"] == ["train"]
    assert summary["num_rows"] == 2
    assert summary["columns"] == ["instruction", "response"]
    assert (artifact.path / "meta.json").exists()


def test_dataset_artifact_save_load_and_summary_for_dataset_dict(tmp_path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_dict({"instruction": ["a"], "response": ["x"]}),
            "test": Dataset.from_dict({"instruction": ["b"], "response": ["y"]}),
        }
    )
    artifact = DatasetArtifact(
        artifact_id="dataset-2",
        name="demo-dataset-dict",
        path=tmp_path / "dataset_dict_artifact",
        producer="test",
    )

    artifact.save(dataset_dict)
    restored = artifact.load()
    summary = artifact.summary()

    assert set(restored.keys()) == {"train", "test"}
    assert summary["hf_object_type"] == "DatasetDict"
    assert summary["splits"] == ["train", "test"]
    assert summary["num_rows"] == {"train": 1, "test": 1}
    assert summary["columns"] == {"train": ["instruction", "response"], "test": ["instruction", "response"]}


def test_artifact_save_meta_writes_record(tmp_path):
    dataset = Dataset.from_dict({"instruction": ["a"], "response": ["x"]})
    artifact = DatasetArtifact(
        artifact_id="dataset-3",
        name="meta-dataset",
        path=tmp_path / "meta_dataset",
        producer="test",
    )

    artifact.save(dataset)
    meta = json.loads((artifact.path / "meta.json").read_text(encoding="utf-8"))

    assert meta["artifact_id"] == "dataset-3"
    assert meta["type"] == "dataset"
    assert meta["meta"]["format"] == "hf_datasets"
