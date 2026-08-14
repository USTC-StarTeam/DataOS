from __future__ import annotations

import json
from pathlib import Path

from dataos_operator import Dataset
from dataos_operator.operators.selector.length_filter import LengthFilter


def test_dataset_usage_example(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_math.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "question": "There are 3 boxes with 4 apples each. How many apples are there?",
                "solution": "3 * 4 = <<3*4=12>>12.\n#### 12",
            }
        ),
        encoding="utf-8",
    )

    dataset = Dataset.from_local(
        raw_path,
        field_map={
            "text": "question",
            "answer": "solution",
        },
        id_prefix="math-demo",
        target_regex=r"####\s*(.+)$",
        target_source_key="answer",
    )

    result = LengthFilter(
        text_key="text",
        min_length=10,
        max_length=100,
    ).run(dataset)

    assert dataset[0] == {
        "text": "There are 3 boxes with 4 apples each. How many apples are there?",
        "answer": "3 * 4 = <<3*4=12>>12.\n#### 12",
        "id": "math-demo-00000",
        "target": "12",
        "metadata": {"source": raw_path.stem},
    }
    assert result.to_records() == dataset.to_records()
