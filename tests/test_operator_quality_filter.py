from __future__ import annotations

from datasets import DatasetDict, Dataset

from dataos.core.sandbox import Sandbox
from dataos.operators.quality_filter import QualityFilterOperator
from operator_test_utils import build_source


def test_quality_filter_heuristic_drops_low_quality_rows(tmp_path):
    source = build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({
            "instruction": ["Explain what TCP is in one paragraph.", "todo"],
            "response": ["TCP is a connection-oriented transport protocol.", "todo"],
        })}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = QualityFilterOperator().run([source], sandbox, min_score=0.6, attach_score_column="score")
    loaded = artifact.load()["train"]
    assert loaded.num_rows == 1
    assert loaded[0]["score"] >= 0.6
    assert artifact.meta["quality_filter_removed_counts"] == {"train": 1}


def test_quality_filter_llm_judge_uses_openai_compatible_path(tmp_path, monkeypatch):
    source = build_source(tmp_path, Dataset.from_dict({"instruction": ["a real prompt"], "response": ["a real answer"]}))
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    monkeypatch.setattr(
        "dataos.operators.quality_filter._judge_with_openai",
        lambda **kwargs: {"keep": True, "score": 0.9, "reason": "ok"},
    )
    artifact = QualityFilterOperator().run(
        [source],
        sandbox,
        mode="llm_judge",
        api_key="test-key",
        model="gpt-test",
        attach_reason_column="judge_reason",
    )
    loaded = artifact.load()
    assert loaded.num_rows == 1
    assert loaded[0]["judge_reason"] == "ok"
