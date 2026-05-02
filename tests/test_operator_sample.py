from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
from dataos.operators.sample import SampleOperator


def _build_source(tmp_path, dataset_obj) -> DatasetArtifact:
    artifact_id = make_artifact_id("dataset")
    path = tmp_path / "src" / artifact_id
    path.mkdir(parents=True)
    artifact = DatasetArtifact(
        artifact_id=artifact_id,
        name="src",
        path=path,
        parent_ids=[],
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact


def test_sample_head_caps_train_split(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict(
            {
                "train": Dataset.from_dict({"x": list(range(10))}),
                "test": Dataset.from_dict({"x": list(range(4))}),
            }
        ),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = SampleOperator().run([source], sandbox, max_train=3)

    loaded = artifact.load()
    assert loaded["train"].num_rows == 3
    assert list(loaded["train"]["x"]) == [0, 1, 2]  # head strategy preserves order
    assert loaded["test"].num_rows == 4  # untouched
    assert artifact.meta["sample_kept_counts"] == {"train": 3, "test": 4}
    assert artifact.meta["sample_removed_counts"] == {"train": 7, "test": 0}


def test_sample_random_is_deterministic(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({"x": list(range(20))})}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    a1 = SampleOperator().run([source], sandbox, max_train=5, strategy="random", seed=7)
    a2 = SampleOperator().run([source], sandbox, max_train=5, strategy="random", seed=7)

    assert list(a1.load()["train"]["x"]) == list(a2.load()["train"]["x"])
    assert a1.load()["train"].num_rows == 5


def test_sample_max_per_split_overrides(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict(
            {
                "train": Dataset.from_dict({"x": list(range(10))}),
                "validation": Dataset.from_dict({"x": list(range(8))}),
            }
        ),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = SampleOperator().run(
        [source],
        sandbox,
        max_per_split={"validation": 2},
        max_train=4,
    )

    loaded = artifact.load()
    assert loaded["train"].num_rows == 4
    assert loaded["validation"].num_rows == 2


def test_sample_requires_caps(tmp_path):
    source = _build_source(
        tmp_path, DatasetDict({"train": Dataset.from_dict({"x": [1, 2]})})
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError):
        SampleOperator().run([source], sandbox)
