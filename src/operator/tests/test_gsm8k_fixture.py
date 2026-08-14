from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATASET = ROOT / "examples" / "datasets" / "gsm8k_target_100.jsonl"
TEST_FIXTURE = ROOT / "tests" / "fixtures" / "gsm8k_target_100.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# 验证 examples 和 tests 两处 GSM8K target 数据各有 100 条。
def test_gsm8k_target_files_have_100_records() -> None:
    example_records = _load_jsonl(EXAMPLE_DATASET)
    fixture_records = _load_jsonl(TEST_FIXTURE)

    assert len(example_records) == 100
    assert len(fixture_records) == 100


# 验证测试 fixture 和 examples 数据集内容保持一致。
def test_gsm8k_target_fixture_matches_example_dataset() -> None:
    assert _load_jsonl(TEST_FIXTURE) == _load_jsonl(EXAMPLE_DATASET)


# 验证 GSM8K target fixture 的字段、编号和 metadata 约定。
def test_gsm8k_target_schema() -> None:
    records = _load_jsonl(TEST_FIXTURE)

    for index, record in enumerate(records):
        assert set(record) == {"id", "text", "answer", "target", "metadata"}
        assert record["id"] == f"gsm8k-train-{index:05d}"
        assert isinstance(record["text"], str)
        assert isinstance(record["answer"], str)
        assert isinstance(record["target"], str)
        assert record["target"]
        assert record["answer"].rstrip().endswith(f"#### {record['target']}")
        assert record["metadata"] == {
            "source": "openai/grade-school-math",
            "dataset": "gsm8k",
            "split": "train",
            "index": index,
        }
