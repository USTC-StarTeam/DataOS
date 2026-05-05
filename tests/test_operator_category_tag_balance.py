from __future__ import annotations

from datasets import DatasetDict, Dataset

from dataos.core.sandbox import Sandbox
from dataos.operators.category_balance import CategoryBalanceOperator
from dataos.operators.category_tag import CategoryTagOperator
from operator_test_utils import build_source


def test_category_tag_assigns_surface_categories(tmp_path):
    source = build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({"instruction": ["Translate this sentence", "Write a short essay"], "response": ["x", "y"]})}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = CategoryTagOperator().run([source], sandbox)
    labels = list(artifact.load()["train"]["dataos_category"])
    assert labels == ["translation", "writing"]


def test_category_balance_downsamples_to_min_bucket(tmp_path):
    source = build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({"instruction": ["a", "b", "c"], "response": ["1", "2", "3"], "dataos_category": ["math", "math", "writing"]})}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = CategoryBalanceOperator().run([source], sandbox)
    loaded = artifact.load()["train"]
    assert loaded.num_rows == 2
    assert artifact.meta["category_balance_output_counts"] == {"math": 1, "writing": 1}
