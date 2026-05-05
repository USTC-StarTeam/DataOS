from __future__ import annotations

from datasets import DatasetDict, Dataset

from dataos.core.sandbox import Sandbox
from dataos.operators.lang_filter import LangFilterOperator
from operator_test_utils import build_source


def test_lang_filter_keeps_requested_language_and_can_annotate(tmp_path):
    source = build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({"instruction": ["Translate this sentence", "把这句话翻译成英文"], "response": ["ok", "好的"]})}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = LangFilterOperator().run([source], sandbox, allowed_languages=["zh"], annotate_column="lang")
    loaded = artifact.load()["train"]
    assert loaded.num_rows == 1
    assert loaded[0]["lang"] == "zh"
    assert artifact.meta["lang_filter_removed_counts"] == {"train": 1}
