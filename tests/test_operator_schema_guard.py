from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from dataos.core.sandbox import Sandbox
from dataos.operators.schema_guard import SchemaGuardOperator
from operator_test_utils import build_source


def test_schema_guard_repairs_common_aliases(tmp_path):
    source = build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({"prompt": [" hi ", ""], "output": [" there ", "x"], "meta": [1, 2]})}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = SchemaGuardOperator().run([source], sandbox, keep_extra_columns=True)
    row = artifact.load()["train"][0]
    assert row["instruction"] == "hi"
    assert row["response"] == "there"
    assert row["input"] == ""
    assert row["meta"] == 1
    assert artifact.meta["schema_guard_dropped_counts"] == {"train": 1}
    assert artifact.parent_ids == [source.artifact_id]
    assert (sandbox.logs_dir / "schema_guard.log").exists()


def test_schema_guard_strict_raises_on_empty_rows(tmp_path):
    source = build_source(tmp_path, Dataset.from_dict({"instruction": ["ok", ""], "response": ["yes", "no"]}))
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError, match="strict"):
        SchemaGuardOperator().run([source], sandbox, mode="strict")
