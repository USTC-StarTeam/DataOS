"""Helpers for exporting DataOS datasets and configs to LLaMA-Factory layouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataos_operator import Dataset
from dataos_operator.core.chat_formats import normalize_chat_messages
from dataos_operator.core.image import normalize_image_list


def export_sharegpt_dataset(
    dataset: Dataset,
    output_dir: str | Path,
    *,
    dataset_name: str,
    split_name: str = "train",
) -> dict[str, Any]:
    """Export a dataset with messages into LLaMA-Factory ShareGPT layout."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{split_name}.jsonl"
    file_path = output_path / file_name
    has_images = any(
        bool(normalize_image_list(record.get("images"), strict=True)) for record in dataset
    )

    with file_path.open("w", encoding="utf-8") as f:
        for record in dataset:
            _, normalized_messages = normalize_chat_messages(record["messages"], strict=True)
            images = normalize_image_list(record.get("images"), strict=True)
            metadata = record.get("metadata", {})
            runtime_metadata = metadata.get("runtime", {}) if isinstance(metadata, dict) else {}
            payload = {
                "messages": normalized_messages,
                "dataos_id": record.get("id"),
                "dataos_domain": (
                    metadata.get("source", metadata.get("domain"))
                    if isinstance(metadata, dict)
                    else None
                ),
                "dataos_reference_loss": runtime_metadata.get("reference_loss")
                if isinstance(runtime_metadata, dict)
                else None,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
            if has_images:
                payload["images"] = images
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    columns = {"messages": "messages"}
    if has_images:
        columns["images"] = "images"
    dataset_info = {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": columns,
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    info_path = output_path / "dataset_info.json"
    info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_info


def write_sft_config(
    output_dir: str | Path,
    *,
    dataset_name: str,
    model_name_or_path: str,
    stage: str = "sft",
    template: str = "default",
    finetuning_type: str = "full",
    cutoff_len: int = 512,
    per_device_train_batch_size: int = 1,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 1.0,
    output_subdir: str = "llamafactory_output",
    extra_fields: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal LLaMA-Factory training YAML next to the exported dataset."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "llamafactory_sft.yaml"
    config = {
        "stage": stage,
        "do_train": True,
        "model_name_or_path": model_name_or_path,
        "dataset_dir": str(output_path),
        "dataset": dataset_name,
        "template": template,
        "finetuning_type": finetuning_type,
        "cutoff_len": cutoff_len,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "output_dir": str(output_path / output_subdir),
    }
    if extra_fields:
        config.update(extra_fields)

    yaml_lines = [f"{key}: {_yaml_scalar(value)}" for key, value in config.items()]
    config_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return config_path


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in (":", "#", "{", "}", "[", "]", "\n")):
        return json.dumps(text, ensure_ascii=False)
    return text
