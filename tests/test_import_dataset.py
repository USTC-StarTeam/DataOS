from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import DatasetDict, load_from_disk

from dataos.core.sandbox import Sandbox
from dataos.operators.import_dataset import ImportDatasetOperator


def _write_alpaca_like_parquet(path: Path) -> None:
    table = pa.table(
        {
            "instruction": ["a" * 5, "b" * 5, "c" * 5, "d" * 5, "", "e" * 5],
            "input": ["", "", "", "", "", ""],
            "output": [
                "answer one",
                "answer two",
                "answer three",
                "answer four",
                "answer five",
                "",
            ],
            "text": ["x"] * 6,
        }
    )
    pq.write_table(table, path)


def test_import_dataset_parquet_with_rename_and_split(tmp_path):
    parquet_path = tmp_path / "alpaca.parquet"
    _write_alpaca_like_parquet(parquet_path)
    output_path = tmp_path / "alpaca-hf"

    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    operator = ImportDatasetOperator()
    spec = {
        "source": "parquet",
        "path": str(parquet_path),
        "rename": {"output": "response"},
        "keep_columns": ["instruction", "input", "response"],
        "required_columns": ["instruction", "response"],
        "train_test_split": {"test_size": 0.25, "seed": 7},
        "max_train_samples": 100,
        "output_path": str(output_path),
        "name": "alpaca-test",
    }
    artifact = operator.run([], sandbox, **spec)

    assert artifact.path == output_path
    assert artifact.meta["splits_after_import"] == ["train", "test"]
    loaded = load_from_disk(str(output_path))
    assert isinstance(loaded, DatasetDict)
    assert set(loaded.keys()) == {"train", "test"}
    for split in loaded.values():
        assert split.column_names == ["instruction", "input", "response"]
        for row in split:
            assert row["instruction"].strip()
            assert row["response"].strip()
    total = loaded["train"].num_rows + loaded["test"].num_rows
    assert total == 4

    log_path = sandbox.logs_dir / "import_dataset.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "source=parquet" in log_text


def test_import_dataset_passes_through_existing_splits(tmp_path):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    pq.write_table(
        pa.table({"prompt": ["p1", "p2", "p3"]}),
        train_path,
    )
    pq.write_table(
        pa.table({"prompt": ["p9"]}),
        test_path,
    )

    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    operator = ImportDatasetOperator()
    output_path = tmp_path / "out"
    artifact = operator.run(
        [],
        sandbox,
        source="parquet",
        path=[str(train_path)],
        output_path=str(output_path),
        max_train_samples=2,
    )
    loaded = load_from_disk(str(output_path))
    assert isinstance(loaded, DatasetDict)
    assert loaded["train"].num_rows == 2

    meta = json.loads((output_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["meta"]["import_spec"]["source"] == "parquet"


def test_dotenv_autoload_does_not_override_existing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "preexisting")
    (tmp_path / ".env").write_text(
        "DASHSCOPE_API_KEY=loaded-from-file\nDASHSCOPE_NEW_KEY=hello\n",
        encoding="utf-8",
    )

    from dataos._env import load_dotenv

    loaded = load_dotenv()

    assert loaded == tmp_path / ".env"

    import os

    assert os.environ["DASHSCOPE_API_KEY"] == "preexisting"
    assert os.environ["DASHSCOPE_NEW_KEY"] == "hello"
