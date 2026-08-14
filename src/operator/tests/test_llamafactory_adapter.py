from __future__ import annotations

import json

from dataos_operator import Dataset
from dataos_operator.adapters import export_sharegpt_dataset, write_sft_config


def test_export_sharegpt_dataset_and_yaml_bundle(tmp_path) -> None:
    dataset = Dataset.from_records(
        [
            {
                "id": "sample-1",
                "messages": [
                    {"role": "user", "content": "<image>\nhello"},
                    {"role": "assistant", "content": "world"},
                ],
                "images": ["images/sample.png"],
                "metadata": {"domain": "gsm8k"},
            }
        ]
    )

    dataset_info = export_sharegpt_dataset(dataset, tmp_path, dataset_name="demo_policy")
    config_path = write_sft_config(
        tmp_path,
        dataset_name="demo_policy",
        model_name_or_path="tiny-model",
    )

    assert dataset_info["demo_policy"]["formatting"] == "sharegpt"
    train_path = tmp_path / "train.jsonl"
    payload = json.loads(train_path.read_text(encoding="utf-8").strip())
    assert payload["dataos_id"] == "sample-1"
    assert payload["dataos_domain"] == "gsm8k"
    assert payload["dataos_reference_loss"] is None
    assert payload["images"] == ["images/sample.png"]
    assert dataset_info["demo_policy"]["columns"]["images"] == "images"
    assert "dataset: demo_policy" in config_path.read_text(encoding="utf-8")
