from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox

_TRACKING_PREFIXES = ("source_", "dataos_")
_TRACKING_FIELDS = {"tags", "meta", "score", "scores"}


def find_latest_dataset(inputs: list[Artifact], operator_name: str) -> DatasetArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError(f"{operator_name} requires at least one DatasetArtifact input")


def find_dataset_artifacts(inputs: list[Artifact], operator_name: str) -> list[DatasetArtifact]:
    datasets = [artifact for artifact in inputs if isinstance(artifact, DatasetArtifact)]
    if not datasets:
        raise ValueError(f"{operator_name} requires at least one DatasetArtifact input")
    return datasets


def ensure_dataset_dict(dataset_obj: Dataset | DatasetDict) -> tuple[DatasetDict, bool]:
    if isinstance(dataset_obj, DatasetDict):
        return dataset_obj, True
    return DatasetDict({"train": dataset_obj}), False


def restore_dataset_type(dataset_dict: DatasetDict, was_dict: bool) -> Dataset | DatasetDict:
    if was_dict:
        return dataset_dict
    return dataset_dict["train"]


def copy_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def copy_row(row: dict[str, Any]) -> dict[str, Any]:
    return copy_value(row)


def infer_first_available(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def require_column(columns: list[str], explicit: str | None, candidates: list[str], operator_name: str) -> str:
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(f"{operator_name}: column '{explicit}' not found")
        return explicit
    resolved = infer_first_available(columns, candidates)
    if resolved is None:
        joined = ", ".join(candidates)
        raise ValueError(f"{operator_name}: could not resolve required column from [{joined}]")
    return resolved


def optional_column(columns: list[str], explicit: str | None, candidates: list[str], operator_name: str) -> str | None:
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(f"{operator_name}: column '{explicit}' not found")
        return explicit
    return infer_first_available(columns, candidates)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def join_text(row: dict[str, Any], columns: list[str]) -> str:
    parts = [normalize_text(row.get(column)) for column in columns]
    return "\n".join(part for part in parts if part)


def count_words(text: str) -> int:
    return len([token for token in text.split() if token])


def detect_dataset_format(columns: list[str], artifact_meta: dict[str, Any] | None = None) -> str:
    artifact_meta = artifact_meta or {}
    if artifact_meta.get("dataset_format") in {"alpaca", "sharegpt", "openai_messages"}:
        return str(artifact_meta["dataset_format"])
    if "conversations" in columns:
        return "sharegpt"
    if "messages" in columns:
        return "openai_messages"
    return "alpaca"


def _extract_from_openai_messages(row: dict[str, Any]) -> tuple[str, str, str]:
    system = normalize_text(row.get("system"))
    prompt_parts: list[str] = []
    answer = ""
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        value = normalize_text(message.get("content") or message.get("value"))
        if role == "system" and value and not system:
            system = value
        elif role in {"human", "user"} and value:
            prompt_parts.append(value)
        elif role in {"assistant", "gpt"} and value:
            answer = value
    return "\n".join(prompt_parts).strip(), answer, system


def _extract_from_sharegpt(row: dict[str, Any]) -> tuple[str, str, str]:
    system = normalize_text(row.get("system"))
    prompt_parts: list[str] = []
    answer = ""
    for message in row.get("conversations") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from") or message.get("role") or "").lower()
        value = normalize_text(message.get("value") or message.get("content"))
        if role in {"human", "user"} and value:
            prompt_parts.append(value)
        elif role in {"assistant", "gpt"} and value:
            answer = value
    return "\n".join(prompt_parts).strip(), answer, system


def extract_text_pair(
    row: dict[str, Any],
    columns: list[str],
    artifact_meta: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    dataset_format = detect_dataset_format(columns, artifact_meta)
    if dataset_format == "openai_messages":
        return _extract_from_openai_messages(row)
    if dataset_format == "sharegpt":
        return _extract_from_sharegpt(row)

    instruction_column = infer_first_available(columns, ["instruction", "prompt", "question"])
    input_column = infer_first_available(columns, ["input", "context", "query"])
    answer_column = infer_first_available(columns, ["output", "response", "answer"])
    instruction = normalize_text(row.get(instruction_column)) if instruction_column else ""
    extra_input = normalize_text(row.get(input_column)) if input_column else ""
    answer = normalize_text(row.get(answer_column)) if answer_column else ""
    prompt = instruction if not extra_input else f"{instruction}\n{extra_input}".strip()
    return prompt, answer, normalize_text(row.get("system"))


def extract_structured_fields(
    row: dict[str, Any],
    columns: list[str],
    artifact_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    if "instruction" in columns and "output" in columns:
        return {
            "instruction": normalize_text(row.get("instruction")),
            "input": normalize_text(row.get("input")),
            "output": normalize_text(row.get("output")),
            "system": normalize_text(row.get("system")),
        }
    prompt, answer, system = extract_text_pair(row, columns, artifact_meta)
    instruction, _, extra_input = prompt.partition("\n")
    return {
        "instruction": instruction.strip(),
        "input": extra_input.strip(),
        "output": answer.strip(),
        "system": system.strip(),
    }


def resolve_text_key(
    row: dict[str, Any],
    key: str,
    columns: list[str],
    artifact_meta: dict[str, Any] | None = None,
) -> str:
    if key in row:
        return normalize_text(row.get(key))
    structured = extract_structured_fields(row, columns, artifact_meta)
    if key in structured:
        return structured[key]
    return ""


def build_alpaca_row(prompt: str, answer: str, system: str = "") -> dict[str, Any]:
    instruction, _, extra_input = prompt.partition("\n")
    row: dict[str, Any] = {
        "instruction": instruction.strip(),
        "input": extra_input.strip(),
        "output": answer.strip(),
    }
    if system.strip():
        row["system"] = system.strip()
    return row


def build_sharegpt_row(prompt: str, answer: str, system: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "conversations": [
            {"from": "human", "value": prompt.strip()},
            {"from": "assistant", "value": answer.strip()},
        ]
    }
    if system.strip():
        row["system"] = system.strip()
    return row


def build_openai_messages_row(prompt: str, answer: str, system: str = "") -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    messages.append({"role": "assistant", "content": answer.strip()})
    return {"messages": messages}


def build_row_for_format(dataset_format: str, prompt: str, answer: str, system: str = "") -> dict[str, Any]:
    if dataset_format == "sharegpt":
        return build_sharegpt_row(prompt, answer, system=system)
    if dataset_format == "openai_messages":
        return build_openai_messages_row(prompt, answer, system=system)
    return build_alpaca_row(prompt, answer, system=system)


def preserve_context_fields(source_row: dict[str, Any], new_row: dict[str, Any], keep_other_columns: bool = False) -> None:
    for key, value in source_row.items():
        should_keep = keep_other_columns or key.startswith(_TRACKING_PREFIXES) or key in _TRACKING_FIELDS
        if should_keep and key not in new_row:
            new_row[key] = copy_value(value)


def save_dataset_artifact(
    *,
    source: DatasetArtifact,
    sandbox: Sandbox,
    operator_name: str,
    suffix: str,
    dataset_obj: Dataset | DatasetDict,
    meta: dict[str, Any],
) -> DatasetArtifact:
    artifact_id = make_artifact_id("dataset")
    artifact = DatasetArtifact(
        artifact_id=artifact_id,
        name=f"{source.name}-{suffix}",
        path=sandbox.artifact_dir("dataset", artifact_id),
        parent_ids=[source.artifact_id],
        producer=operator_name,
        meta=meta,
    )
    artifact.save(dataset_obj)
    return artifact


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
