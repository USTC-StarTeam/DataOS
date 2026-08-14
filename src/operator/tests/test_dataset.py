from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from dataos_operator import Dataset
from dataos_operator.operators.selector.length_filter import LengthFilter

ROOT = Path(__file__).resolve().parents[1]
GSM8K_FIXTURE = ROOT / "tests" / "fixtures" / "gsm8k_target_100.jsonl"


def test_dataos_dataset_from_local_jsonl_can_feed_operator() -> None:
    dataset = Dataset.from_local(GSM8K_FIXTURE, limit=2)
    operator = LengthFilter(text_key="text", min_length=120, max_length=160)

    result = operator.run(dataset)

    assert len(dataset) == 2
    assert dataset.metadata["source"] == GSM8K_FIXTURE.stem
    assert dataset[0]["metadata"]["source"] == GSM8K_FIXTURE.stem
    assert result.metadata["input_records"] == 2
    assert result.metadata["output_records"] == 1
    assert result.to_records() == [dict(dataset[0])]


def test_dataos_dataset_from_local_can_map_fields_and_extract_target(tmp_path: Path) -> None:
    source_path = tmp_path / "raw_gsm8k.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "question": "What is 1 + 1?",
                "answer": "1 + 1 = <<1+1=2>>2.\n#### 2",
            }
        ),
        encoding="utf-8",
    )

    dataset = Dataset.from_local(
        source_path,
        field_map={
            "text": "question",
            "answer": "answer",
        },
        id_prefix="sample",
        target_regex=r"####\s*(.+)$",
    )

    assert dataset[0] == {
        "text": "What is 1 + 1?",
        "answer": "1 + 1 = <<1+1=2>>2.\n#### 2",
        "id": "sample-00000",
        "target": "2",
        "metadata": {"source": "raw_gsm8k"},
    }


def test_dataos_dataset_from_source_accepts_local_path() -> None:
    dataset = Dataset.from_source(GSM8K_FIXTURE, limit=1)

    assert len(dataset) == 1
    assert dataset[0]["id"] == "gsm8k-train-00000"
    assert dataset[0]["metadata"]["source"] == GSM8K_FIXTURE.stem


def test_dataos_dataset_from_hf_uses_datasets_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
        assert args == ("openai/gsm8k", "main")
        assert kwargs == {
            "split": "train[:1]",
            "streaming": False,
        }
        return [
            {
                "question": "What is 2 + 3?",
                "answer": "2 + 3 = <<2+3=5>>5.\n#### 5",
            }
        ]

    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    dataset = Dataset.from_hf(
        "openai/gsm8k",
        name="main",
        split="train[:1]",
        field_map={
            "text": "question",
            "answer": "answer",
        },
        target_regex=r"####\s*(.+)$",
    )

    assert dataset[0] == {
        "text": "What is 2 + 3?",
        "answer": "2 + 3 = <<2+3=5>>5.\n#### 5",
        "id": "openai-gsm8k-train-1-00000",
        "target": "5",
        "metadata": {
            "source": "gsm8k",
            "path": "openai/gsm8k",
            "split": "train[:1]",
            "loader": "huggingface",
            "name": "main",
            "index": 0,
        },
    }
    assert dataset.metadata == {
        "source": "gsm8k",
        "path": "openai/gsm8k",
        "split": "train[:1]",
        "loader": "huggingface",
        "name": "main",
        "records": 1,
        "streaming": False,
    }


def test_dataos_dataset_saves_local_jsonl_json_and_csv(tmp_path: Path) -> None:
    dataset = Dataset.from_records(
        [
            {
                "id": "sample-1",
                "text": "hello",
                "metadata": {"source": "unit"},
            }
        ]
    )
    jsonl_path = dataset.save_local(tmp_path / "dataset.jsonl")
    json_path = dataset.save_local(tmp_path / "dataset.json")
    csv_path = dataset.save_local(tmp_path / "dataset.csv")

    loaded_jsonl = Dataset.from_local(jsonl_path)[0]
    loaded_json = Dataset.from_local(json_path)[0]
    assert loaded_jsonl["id"] == "sample-1"
    assert loaded_jsonl["text"] == "hello"
    assert loaded_jsonl["metadata"]["source"] == "dataset"
    assert loaded_json["metadata"]["source"] == "dataset"
    with csv_path.open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows == [
        {
            "id": "sample-1",
            "text": "hello",
            "metadata": '{"source": "unit"}',
        }
    ]


def test_dataos_dataset_save_local_can_prevent_overwrite(tmp_path: Path) -> None:
    dataset = Dataset.from_records([{"text": "hello"}])
    path = dataset.save_local(tmp_path / "dataset.jsonl")

    with pytest.raises(FileExistsError):
        dataset.save_local(path, overwrite=False)


def test_dataos_dataset_save_local_accepts_explicit_format(tmp_path: Path) -> None:
    dataset = Dataset.from_records([{"text": "hello"}])
    path = dataset.save_local(tmp_path / "dataset_without_suffix", file_format="ndjson")

    assert path.read_text(encoding="utf-8") == '{"text": "hello"}\n'


def test_dataos_dataset_pushes_to_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    pushed: dict[str, Any] = {}

    class FakeHFDataset:
        def __init__(self, records: list[dict[str, str]]) -> None:
            self.records = records

        @classmethod
        def from_list(cls, records: list[dict[str, str]]) -> FakeHFDataset:
            return cls(records)

        def push_to_hub(self, *args: Any, **kwargs: Any) -> str:
            pushed["records"] = self.records
            pushed["args"] = args
            pushed["kwargs"] = kwargs
            return "commit-info"

    fake_datasets = ModuleType("datasets")
    fake_datasets.Dataset = FakeHFDataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    dataset = Dataset.from_records([{"text": "hello"}])
    result = dataset.push_to_hf(
        "org/repo",
        split="validation",
        config_name="main",
        private=True,
        commit_message="upload dataset",
    )

    assert result == "commit-info"
    assert pushed == {
        "records": [{"text": "hello"}],
        "args": ("org/repo",),
        "kwargs": {
            "split": "validation",
            "config_name": "main",
            "token": None,
            "private": True,
            "commit_message": "upload dataset",
            "create_pr": False,
        },
    }


def test_dataos_dataset_from_source_accepts_hf_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = lambda *args, **kwargs: [{"text": "hello"}]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    dataset = Dataset.from_source("hf://sample/repo", limit=1)

    assert dataset[0]["text"] == "hello"
    assert dataset[0]["metadata"]["source"] == "repo"
    assert dataset[0]["metadata"]["loader"] == "huggingface"
