from __future__ import annotations

import json

import pytest
from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
from dataos.operators.less import LessSelectOperator


def _build_artifact(tmp_path, name: str, dataset_obj) -> DatasetArtifact:
    artifact_id = make_artifact_id("dataset")
    path = tmp_path / "src" / artifact_id
    path.mkdir(parents=True)
    artifact = DatasetArtifact(
        artifact_id=artifact_id,
        name=name,
        path=path,
        parent_ids=[],
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact


def _pool(n: int = 20) -> Dataset:
    return Dataset.from_dict(
        {
            "instruction": [f"Q-{i}" for i in range(n)],
            "response": [f"A-{i}" for i in range(n)],
        }
    )


def _target(n: int = 4) -> Dataset:
    return Dataset.from_dict(
        {
            "instruction": [f"target-{i}" for i in range(n)],
            "response": [f"target-resp-{i}" for i in range(n)],
        }
    )


def _make_pair(tmp_path):
    pool = _build_artifact(tmp_path, "pool", _pool())
    target = _build_artifact(tmp_path, "target", _target())
    return pool, target


def test_less_stub_selects_top_k_with_lineage_and_manifest(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = LessSelectOperator().run(
        [target, pool],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        top_k=5,
        mode="stub",
        seed=7,
    )

    loaded = artifact.load()
    assert isinstance(loaded, Dataset)
    assert loaded.num_rows == 5
    assert artifact.parent_ids == [pool.artifact_id, target.artifact_id]
    assert artifact.meta["less_mode"] == "stub"
    assert artifact.meta["less_pool_count"] == 20
    assert artifact.meta["less_selected_count"] == 5
    assert artifact.meta["less_target_artifact_id"] == target.artifact_id
    stage_names = [stage["name"] for stage in artifact.meta["less_stages"]]
    assert stage_names == ["warmup", "train_grads", "val_grads", "select"]
    assert all(stage["cache_status"] == "skipped" for stage in artifact.meta["less_stages"])

    manifest = json.loads((artifact.path / "selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_count"] == 5
    assert len(manifest["selected_indices"]) == 5
    assert manifest["pool_artifact_id"] == pool.artifact_id
    assert manifest["target_artifact_id"] == target.artifact_id

    assert (sandbox.logs_dir / "less.log").exists()
    for stage in ("warmup", "train_grads", "val_grads", "select"):
        assert (sandbox.logs_dir / f"less_{stage}.log").exists()


def test_less_stub_top_fraction(tmp_path):
    pool = _build_artifact(tmp_path, "pool", _pool(100))
    target = _build_artifact(tmp_path, "target", _target())
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = LessSelectOperator().run(
        [target, pool],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_fraction=0.1,
        mode="stub",
    )

    assert artifact.load().num_rows == 10
    assert artifact.meta["less_top_fraction"] == 0.1
    assert artifact.meta["less_top_k"] is None


def test_less_preserves_other_splits_when_pool_is_dict(tmp_path):
    pool = _build_artifact(
        tmp_path,
        "pool",
        DatasetDict(
            {
                "train": _pool(20),
                "test": Dataset.from_dict({"instruction": ["t1"], "response": ["r1"]}),
            }
        ),
    )
    target = _build_artifact(tmp_path, "target", _target())
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = LessSelectOperator().run(
        [target, pool],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=5,
        mode="stub",
    )

    loaded = artifact.load()
    assert isinstance(loaded, DatasetDict)
    assert loaded["train"].num_rows == 5
    assert loaded["test"].num_rows == 1


def test_less_stub_is_deterministic_with_seed(tmp_path):
    pool = _build_artifact(tmp_path, "pool", _pool(50))
    target = _build_artifact(tmp_path, "target", _target())

    sandbox_a = Sandbox(tmp_path / "sandbox_a")
    sandbox_a.prepare()
    artifact_a = LessSelectOperator().run(
        [target, pool],
        sandbox_a,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=10,
        mode="stub",
        seed=42,
    )
    sandbox_b = Sandbox(tmp_path / "sandbox_b")
    sandbox_b.prepare()
    artifact_b = LessSelectOperator().run(
        [target, pool],
        sandbox_b,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=10,
        mode="stub",
        seed=42,
    )

    indices_a = json.loads((artifact_a.path / "selection_manifest.json").read_text(encoding="utf-8"))[
        "selected_indices"
    ]
    indices_b = json.loads((artifact_b.path / "selection_manifest.json").read_text(encoding="utf-8"))[
        "selected_indices"
    ]
    assert indices_a == indices_b


def test_less_pool_position_in_inputs_does_not_matter(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    # Order: pool first, target last (inverse of the typical pattern).
    artifact = LessSelectOperator().run(
        [pool, target],
        sandbox,
        target_artifact_id=target.artifact_id,
        base_model="m",
        top_k=5,
        mode="stub",
    )

    assert artifact.parent_ids == [pool.artifact_id, target.artifact_id]


def test_less_requires_target_artifact_id(tmp_path):
    pool, _ = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="target_artifact_id"):
        LessSelectOperator().run([pool], sandbox, base_model="m", top_k=5, mode="stub")


def test_less_target_artifact_id_not_in_inputs(tmp_path):
    pool, _ = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="not found in inputs"):
        LessSelectOperator().run(
            [pool],
            sandbox,
            target_artifact_id="dataset-does-not-exist",
            base_model="m",
            top_k=5,
            mode="stub",
        )


def test_less_pool_must_differ_from_target(tmp_path):
    only = _build_artifact(tmp_path, "only", _pool())
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="distinct from the target"):
        LessSelectOperator().run(
            [only],
            sandbox,
            target_artifact_id=only.artifact_id,
            base_model="m",
            top_k=5,
            mode="stub",
        )


def test_less_pool_missing_required_columns(tmp_path):
    pool = _build_artifact(tmp_path, "pool", Dataset.from_dict({"text": ["a", "b", "c"]}))
    target = _build_artifact(tmp_path, "target", _target())
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="missing required columns"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="stub",
        )


def test_less_target_missing_required_columns(tmp_path):
    pool = _build_artifact(tmp_path, "pool", _pool())
    target = _build_artifact(tmp_path, "target", Dataset.from_dict({"foo": ["x"]}))
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="target dataset is missing required columns"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="stub",
        )


def test_less_top_k_or_fraction_required(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="top_k.*top_fraction"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            mode="stub",
        )


def test_less_top_k_and_fraction_mutually_exclusive(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="mutually exclusive"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            top_fraction=0.5,
            mode="stub",
        )


def test_less_top_k_exceeds_pool_size_raises(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="exceeds pool size"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=999,
            mode="stub",
        )


def test_less_real_mode_requires_repo_path(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="less_repo_path"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="less",
        )


def test_less_unknown_param_rejected(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="unsupported params"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="stub",
            mystery_kwarg=True,
        )


def test_less_invalid_mode_rejected(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="'mode' must be one of"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="bogus",
        )


def test_less_real_mode_not_yet_implemented(tmp_path):
    pool, target = _make_pair(tmp_path)
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(NotImplementedError, match="commit 2"):
        LessSelectOperator().run(
            [target, pool],
            sandbox,
            target_artifact_id=target.artifact_id,
            base_model="m",
            top_k=5,
            mode="less",
            less_repo_path="/some/path",
        )


def test_less_registered_in_default_registry():
    from dataos.operators import build_default_registry

    registry = build_default_registry()
    operator = registry.get("less")
    assert isinstance(operator, LessSelectOperator)
