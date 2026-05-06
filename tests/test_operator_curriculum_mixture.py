from __future__ import annotations

from datasets import DatasetDict, Dataset

from dataos.core.sandbox import Sandbox
from dataos.operators.curriculum_sort import CurriculumSortOperator
from dataos.operators.mixture import MixtureOperator
from operator_test_utils import build_source


def test_curriculum_sort_orders_easy_to_hard(tmp_path):
    source = build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({"instruction": ["short", "please solve this problem step by step and explain carefully"], "response": ["ok", "longer answer here"]})}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = CurriculumSortOperator().run([source], sandbox, order="easy_to_hard")
    loaded = artifact.load()["train"]
    assert loaded[0]["instruction"] == "short"
    assert "dataos_curriculum_score" in loaded.column_names


def test_mixture_combines_multiple_parents_with_provenance(tmp_path):
    left = build_source(tmp_path, DatasetDict({"train": Dataset.from_dict({"instruction": ["left-1", "left-2"], "response": ["a", "b"]})}), name="left")
    right = build_source(tmp_path, DatasetDict({"train": Dataset.from_dict({"instruction": ["right-1", "right-2"], "response": ["c", "d"]})}), name="right")
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = MixtureOperator().run([left, right], sandbox, max_rows=3, weights={left.artifact_id: 1, right.artifact_id: 2})
    loaded = artifact.load()["train"]
    assert loaded.num_rows == 3
    assert "dataos_source_artifact" in loaded.column_names
    assert artifact.parent_ids == [left.artifact_id, right.artifact_id]
