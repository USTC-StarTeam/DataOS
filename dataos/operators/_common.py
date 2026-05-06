from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact


def find_latest_dataset(inputs: list[Artifact], operator_name: str) -> DatasetArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError(f"{operator_name} requires at least one DatasetArtifact input")


def collect_dataset_artifacts(inputs: list[Artifact]) -> list[DatasetArtifact]:
    return [artifact for artifact in inputs if isinstance(artifact, DatasetArtifact)]


def ensure_dataset_dict(dataset_obj: Dataset | DatasetDict) -> tuple[DatasetDict, bool]:
    if isinstance(dataset_obj, DatasetDict):
        return dataset_obj, True
    return DatasetDict({"train": dataset_obj}), False


def restore_dataset_type(dataset_obj: DatasetDict, was_dict: bool) -> Dataset | DatasetDict:
    if was_dict:
        return dataset_obj
    return dataset_obj["train"]


def infer_text_columns(dataset: Dataset, preferred: list[str] | None = None) -> list[str]:
    preferred = preferred or []
    columns = list(dataset.column_names)
    selected: list[str] = []
    for name in preferred:
        if name in columns and name not in selected:
            selected.append(name)
    if selected:
        return selected

    for name in columns:
        sample = None
        upper = min(dataset.num_rows, 10)
        for index in range(upper):
            value = dataset[index].get(name)
            if value is not None:
                sample = value
                break
        if isinstance(sample, str):
            selected.append(name)
    return selected


def normalize_text(value: Any, *, strip: bool = True, collapse_whitespace: bool = True) -> str:
    if value is None:
        return ""
    text = str(value)
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    if strip:
        text = text.strip()
    return text


def join_text_columns(row: dict[str, Any], columns: Iterable[str]) -> str:
    parts: list[str] = []
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                parts.append(text)
        else:
            parts.append(str(value))
    return "\n".join(parts)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def count_chars(text: str) -> int:
    return len(text)


def detect_language_label(text: str, *, min_fraction: float = 0.6) -> str:
    han = 0
    latin = 0
    other_letters = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            han += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
        elif ch.isalpha():
            other_letters += 1
    total = han + latin + other_letters
    if total == 0:
        return "other"
    ratios = {
        "zh": han / total,
        "en": latin / total,
        "other": other_letters / total,
    }
    label, ratio = max(ratios.items(), key=lambda item: item[1])
    active = sum(1 for value in ratios.values() if value > 0.15)
    if active >= 2 and ratio < min_fraction:
        return "mixed"
    if ratio < min_fraction:
        return "mixed"
    return label


def round_robin_merge(records_by_key: dict[str, list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    keys = [key for key, records in records_by_key.items() if records]
    indices = {key: 0 for key in keys}
    output: list[dict[str, Any]] = []
    while len(output) < limit and keys:
        next_keys: list[str] = []
        for key in keys:
            index = indices[key]
            rows = records_by_key[key]
            if index < len(rows):
                output.append(rows[index])
                indices[key] = index + 1
                if len(output) >= limit:
                    break
            if indices[key] < len(rows):
                next_keys.append(key)
        keys = next_keys
    return output
