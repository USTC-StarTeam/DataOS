from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
from dataos.operators.format_convert import FormatConvertOperator


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


def test_alpaca_to_messages_with_system_prompt(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "instruction": ["sum two numbers", "translate"],
                        "input": ["2 and 3", ""],
                        "output": ["5", "bonjour"],
                    }
                )
            }
        ),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = FormatConvertOperator().run(
        [source], sandbox, target="messages", system_prompt="You are helpful."
    )

    loaded = artifact.load()
    assert loaded["train"].column_names == ["messages"]
    first = loaded["train"][0]["messages"]
    assert first[0] == {"role": "system", "content": "You are helpful."}
    assert first[1]["role"] == "user"
    assert "sum two numbers" in first[1]["content"]
    assert "2 and 3" in first[1]["content"]
    assert first[2] == {"role": "assistant", "content": "5"}

    second = loaded["train"][1]["messages"]
    assert second[1]["content"] == "translate"  # empty input dropped
    assert artifact.meta["format_target"] == "messages"
    assert artifact.meta["format_source"] == {"train": "alpaca"}


def test_alpaca_with_response_column_alias(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "instruction": ["a"],
                        "input": [""],
                        "response": ["b"],
                    }
                )
            }
        ),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = FormatConvertOperator().run([source], sandbox, target="messages")

    msgs = artifact.load()["train"][0]["messages"]
    assert msgs[-1]["content"] == "b"


def test_sharegpt_to_messages(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "conversations": [
                            [
                                {"from": "human", "value": "hi"},
                                {"from": "gpt", "value": "hello"},
                            ]
                        ]
                    }
                )
            }
        ),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = FormatConvertOperator().run([source], sandbox, target="messages")

    msgs = artifact.load()["train"][0]["messages"]
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_messages_to_alpaca(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "messages": [
                            [
                                {"role": "system", "content": "be terse"},
                                {"role": "user", "content": "what is 2+2?"},
                                {"role": "assistant", "content": "4"},
                            ]
                        ]
                    }
                )
            }
        ),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = FormatConvertOperator().run([source], sandbox, target="alpaca")

    row = artifact.load()["train"][0]
    assert row["instruction"].startswith("be terse")
    assert "what is 2+2?" in row["instruction"]
    assert row["output"] == "4"


def test_dataset_dict_input_preserves_splits(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"instruction": ["a"], "input": [""], "output": ["b"]}
                ),
                "test": Dataset.from_dict(
                    {"instruction": ["c"], "input": [""], "output": ["d"]}
                ),
            }
        ),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    artifact = FormatConvertOperator().run([source], sandbox, target="messages")

    loaded = artifact.load()
    assert set(loaded.keys()) == {"train", "test"}
    assert artifact.meta["format_source"] == {"train": "alpaca", "test": "alpaca"}


def test_unknown_target_raises(tmp_path):
    source = _build_source(
        tmp_path,
        DatasetDict({"train": Dataset.from_dict({"instruction": ["a"], "output": ["b"]})}),
    )
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    with pytest.raises(ValueError):
        FormatConvertOperator().run([source], sandbox, target="parquet")
