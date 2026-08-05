"""Shared helpers for policy-runtime demos."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dataos_operator import Dataset
from dataos_operator.core import Record, set_field
from dataos_operator.operators.mapper.openai_message_mapper import OpenAIMessageMapper
from dataos_operator.runtime import BatchView
from dataos_operator.runtime.text_backend import TextModelBackend

ROOT = Path(__file__).resolve().parents[1]
GSM8K_PATH = ROOT / "dataset" / "gsm8k" / "gsm8k.jsonl"
ALPACA_PATH = ROOT / "dataset" / "alpaca" / "alpaca.jsonl"
OUTPUT_DIR = ROOT / "examples" / "outputs"
DEFAULT_MODEL = os.environ.get("DATAOS_OPERATOR_DEMO_MODEL", "sshleifer/tiny-gpt2")
DEFAULT_DEVICE = os.environ.get("DATAOS_OPERATOR_DEMO_DEVICE")
LOCAL_FILES_ONLY = os.environ.get("DATAOS_OPERATOR_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}
DEFAULT_GRAD_FILTERS = tuple(
    token.strip()
    for token in os.environ.get("DATAOS_OPERATOR_GRAD_FILTERS", "").split(",")
    if token.strip()
)


def load_demo_datasets(
    *,
    gsm8k_limit: int,
    alpaca_limit: int,
) -> tuple[Dataset, Dataset]:
    gsm8k = Dataset.from_local(
        GSM8K_PATH,
        source_name="gsm8k",
        limit=gsm8k_limit,
        id_prefix="gsm8k",
    )
    alpaca = Dataset.from_local(
        ALPACA_PATH,
        source_name="alpaca",
        limit=alpaca_limit,
        id_prefix="alpaca",
    )
    return _tag_domain(gsm8k, "gsm8k"), _tag_domain(alpaca, "alpaca")


def build_backend(
    *,
    model_name: str = DEFAULT_MODEL,
    max_length: int = 192,
    max_new_tokens: int = 32,
    projection_filters: tuple[str, ...] = DEFAULT_GRAD_FILTERS,
) -> TextModelBackend:
    return TextModelBackend(
        model_name,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        parameter_filters=projection_filters,
        device=DEFAULT_DEVICE,
        local_files_only=LOCAL_FILES_ONLY,
    )


def merge_datasets(*datasets: Dataset, source_name: str = "merged_demo") -> Dataset:
    records: list[Record] = []
    for dataset in datasets:
        records.extend(dataset.to_records())
    return Dataset.from_records(
        records,
        metadata={
            "source": source_name,
            "records": len(records),
            "components": [dataset.metadata.get("source") for dataset in datasets],
        },
    )


def add_losses(
    dataset: Dataset,
    losses: list[float],
    *,
    loss_key: str = "metadata.runtime.loss",
    initial_loss_key: str | None = None,
) -> Dataset:
    records: list[Record] = []
    for record, loss in zip(dataset, losses, strict=True):
        updated = set_field(record, loss_key, float(loss))
        if initial_loss_key is not None:
            updated = set_field(updated, initial_loss_key, float(loss))
        records.append(updated)
    return Dataset.from_records(records, metadata=dict(dataset.metadata))


def add_embeddings(
    dataset: Dataset,
    embeddings: list[list[float]],
    *,
    embedding_key: str = "metadata.embedding.vector",
) -> Dataset:
    records: list[Record] = []
    for record, embedding in zip(dataset, embeddings, strict=True):
        updated = set_field(record, embedding_key, [float(value) for value in embedding])
        records.append(updated)
    return Dataset.from_records(records, metadata=dict(dataset.metadata))


def dataset_from_indices(
    dataset: Dataset,
    indices: list[int],
    *,
    label: str,
) -> Dataset:
    return Dataset.from_records(
        [dict(dataset[index]) for index in indices],
        metadata={
            "source": dataset.metadata.get("source", "demo"),
            "records": len(indices),
            "selection": label,
            "selected_indices": list(indices),
        },
    )


def dataset_to_messages(gsm8k: Dataset, alpaca: Dataset) -> Dataset:
    gsm8k_messages = OpenAIMessageMapper(
        source_format="qa",
        query_key="question",
        answer_key="answer",
    ).run(gsm8k)
    alpaca_messages = OpenAIMessageMapper(source_format="alpaca").run(alpaca)
    return merge_datasets(gsm8k_messages, alpaca_messages, source_name="messages_demo")


def chunk_as_single_batch(dataset: Dataset) -> BatchView:
    return BatchView(
        records=dataset.to_records(),
        indices=list(range(len(dataset))),
        domain_ids=[str(record.get("metadata", {}).get("domain", "unknown")) for record in dataset],
    )


def ensure_output_dir(name: str) -> Path:
    output_dir = OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dataset(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_local(path)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def short_records(dataset: Dataset, *, count: int = 2) -> list[dict[str, Any]]:
    return [_prune_record(dict(record)) for record in dataset[:count]]


def runtime_loss_schedule(base_losses: list[float], domains: list[str], step: int) -> list[float]:
    adjusted: list[float] = []
    for index, (loss, domain) in enumerate(zip(base_losses, domains, strict=True)):
        domain_shift = -0.10 * step if domain == "gsm8k" else -0.04 * step
        position_shift = 0.02 * ((index % 4) - 1.5) * step
        adjusted.append(max(float(loss) + domain_shift + position_shift, 1e-4))
    return adjusted


def _tag_domain(dataset: Dataset, domain: str) -> Dataset:
    records: list[Record] = []
    for record in dataset:
        updated = dict(record)
        metadata = dict(updated.get("metadata", {}))
        metadata["domain"] = domain
        metadata["source"] = domain
        updated["metadata"] = metadata
        records.append(updated)
    return Dataset.from_records(records, metadata=dict(dataset.metadata))


def _prune_record(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    text_value = updated.get("text")
    if isinstance(text_value, str) and len(text_value) > 280:
        updated["text"] = text_value[:277] + "..."

    metadata = updated.get("metadata")
    if isinstance(metadata, dict):
        metadata_copy = dict(metadata)
        embedding_meta = metadata_copy.get("embedding")
        if isinstance(embedding_meta, dict) and "vector" in embedding_meta:
            embedding_copy = dict(embedding_meta)
            embedding_copy["vector"] = f"<{len(embedding_copy['vector'])} dims>"
            metadata_copy["embedding"] = embedding_copy
        runtime_meta = metadata_copy.get("runtime")
        if isinstance(runtime_meta, dict):
            runtime_copy = dict(runtime_meta)
            for key, value in list(runtime_copy.items()):
                if isinstance(value, list) and len(value) > 8:
                    runtime_copy[key] = f"<{len(value)} values>"
            metadata_copy["runtime"] = runtime_copy
        updated["metadata"] = metadata_copy
    return updated
