from __future__ import annotations

import pytest
from datasets import DatasetDict, Dataset

from dataos.core.sandbox import Sandbox
from dataos.operators.length_filter import LengthFilterOperator
from operator_test_utils import build_source


def test_length_filter_keeps_rows_in_range(tmp_path):
    source = build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({"instruction": ["one two", "one two three four five"], "response": ["x", "y"]})}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = LengthFilterOperator().run([source], sandbox, min_length=2, max_length=4, unit="words")
    loaded = artifact.load()["train"]
    assert loaded.num_rows == 1
    assert loaded[0]["instruction"] == "one two"
    assert artifact.meta["length_filter_removed_counts"] == {"train": 1}


def test_length_filter_rejects_bad_unit(tmp_path):
    source = build_source(tmp_path, Dataset.from_dict({"instruction": ["a"], "response": ["b"]}))
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    with pytest.raises(ValueError):
        LengthFilterOperator().run([source], sandbox, unit="tokens")
