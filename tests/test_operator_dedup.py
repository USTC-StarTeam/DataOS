from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
from dataos.operators.dedup import DedupOperator


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


def test_dedup_drops_duplicates_per_split(tmp_path):
    train = Dataset.from_dict(
        {
            "instruction": ["Hello", "hello", "Hi there", "Hello "],
            "response": ["a", "b", "c", "d"],
        }
    )
    test = Dataset.from_dict(
        {"instruction": ["q1", "q1", "q2"], "response": ["x", "y", "z"]}
    )
    source = _build_source(tmp_path, DatasetDict({"train": train, "test": test}))
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = DedupOperator().run(
        [source],
        sandbox,
        keys=["instruction"],
    )

    loaded = artifact.load()
    assert loaded["train"].num_rows == 2
    assert loaded["test"].num_rows == 2
    assert artifact.meta["dedup_removed_counts"] == {"train": 2, "test": 1}
    assert artifact.parent_ids == [source.artifact_id]
    assert (sandbox.logs_dir / "dedup.log").exists()


def test_dedup_respects_normalize_false(tmp_path):
    dataset = DatasetDict(
        {"train": Dataset.from_dict({"q": ["A", "a", "A "], "r": [1, 2, 3]})}
    )
    source = _build_source(tmp_path, dataset)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = DedupOperator().run(
        [source], sandbox, keys=["q"], normalize=False
    )

    loaded = artifact.load()
    assert loaded["train"].num_rows == 3
    assert artifact.meta["dedup_normalize"] is False


def test_dedup_cross_split_when_per_split_false(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict(
            {
                "train": Dataset.from_dict({"q": ["a", "b"], "r": [1, 2]}),
                "test": Dataset.from_dict({"q": ["a", "c"], "r": [3, 4]}),
            }
        ),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = DedupOperator().run(
        [source], sandbox, keys=["q"], per_split=False
    )

    loaded = artifact.load()
    assert loaded["train"].num_rows == 2
    assert loaded["test"].num_rows == 1  # 'a' already seen in train


def test_dedup_requires_keys(tmp_path):
    source = _build_source(
        tmp_path, DatasetDict({"train": Dataset.from_dict({"q": ["a"]})})
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    try:
        DedupOperator().run([source], sandbox)
    except ValueError as exc:
        assert "keys" in str(exc)
    else:
        raise AssertionError("expected ValueError when keys missing")
