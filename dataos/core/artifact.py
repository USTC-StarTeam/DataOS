from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

ARTIFACT_TYPES = {"dataset", "model", "eval", "result", "experiment"}


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def make_artifact_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _dataset_columns(dataset_obj: Dataset | DatasetDict) -> dict[str, list[str]] | list[str]:
    if isinstance(dataset_obj, DatasetDict):
        return {split: list(split_dataset.column_names) for split, split_dataset in dataset_obj.items()}
    return list(dataset_obj.column_names)


def _dataset_features(dataset_obj: Dataset | DatasetDict) -> dict[str, Any]:
    if isinstance(dataset_obj, DatasetDict):
        return {
            split: _jsonable(split_dataset.features)
            for split, split_dataset in dataset_obj.items()
        }
    return _jsonable(dataset_obj.features)


def _dataset_num_rows(dataset_obj: Dataset | DatasetDict) -> int | dict[str, int]:
    if isinstance(dataset_obj, DatasetDict):
        return {split: split_dataset.num_rows for split, split_dataset in dataset_obj.items()}
    return dataset_obj.num_rows


def _dataset_fingerprint(dataset_obj: Dataset | DatasetDict) -> str | None:
    if isinstance(dataset_obj, DatasetDict):
        fingerprints = {
            split: getattr(split_dataset, "_fingerprint", None)
            for split, split_dataset in dataset_obj.items()
        }
        return json.dumps(fingerprints, sort_keys=True)
    return getattr(dataset_obj, "_fingerprint", None)


def select_working_dataset(dataset_obj: Dataset | DatasetDict) -> tuple[Dataset, str, list[str]]:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj, "train", ["train"]
    split_names = list(dataset_obj.keys())
    if "train" in dataset_obj:
        return dataset_obj["train"], "train", split_names
    if len(split_names) == 1:
        split_name = split_names[0]
        return dataset_obj[split_name], split_name, split_names
    raise ValueError(
        "DatasetArtifact source must provide a 'train' split, or exactly one split if 'train' is absent"
    )


@dataclass(slots=True)
class Artifact(ABC):
    artifact_id: str
    name: str
    path: Path | str
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    parent_ids: list[str] = field(default_factory=list)
    producer: str = "unknown"
    type: str = field(init=False)

    def __post_init__(self) -> None:
        if not hasattr(self, "type"):
            raise ValueError("Artifact subclasses must assign a fixed type before Artifact.__post_init__")
        self.path = Path(self.path)
        if self.type not in ARTIFACT_TYPES:
            raise ValueError(f"Unsupported artifact type: {self.type}")
        if self.path.exists() and not self.path.is_dir():
            raise ValueError(f"Artifact path must be a directory: {self.path}")

    def save_meta(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        meta_path = self.path / "meta.json"
        meta_path.write_text(
            json.dumps(self.to_record(), indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.type,
            "path": str(self.path),
            "meta": _jsonable(self.meta),
            "created_at": self.created_at,
            "parent_ids": self.parent_ids,
            "producer": self.producer,
        }

    def exists(self) -> bool:
        return self.path.exists()


@dataclass(slots=True)
class DatasetArtifact(Artifact):
    source: str | Path | None = None

    def __post_init__(self) -> None:
        self.type = "dataset"
        Artifact.__post_init__(self)
        if self.source is not None:
            self.source = str(self.source)
        self.meta.setdefault("format", "hf_datasets")
        if self.meta["format"] != "hf_datasets":
            raise ValueError("DatasetArtifact only supports Hugging Face datasets format")

    def _inspect_dataset(self, dataset_obj: Dataset | DatasetDict) -> dict[str, Any]:
        if isinstance(dataset_obj, Dataset):
            splits = ["train"]
            object_type = "Dataset"
        elif isinstance(dataset_obj, DatasetDict):
            splits = list(dataset_obj.keys())
            object_type = "DatasetDict"
        else:
            raise TypeError(f"Unsupported dataset object type: {type(dataset_obj)!r}")
        return {
            "hf_object_type": object_type,
            "num_rows": _dataset_num_rows(dataset_obj),
            "splits": splits,
            "features": _dataset_features(dataset_obj),
            "fingerprint": _dataset_fingerprint(dataset_obj),
            "format": "hf_datasets",
            "columns": _dataset_columns(dataset_obj),
        }

    def refresh_metadata_from_disk(self) -> None:
        dataset_obj = self.load()
        inspected = self._inspect_dataset(dataset_obj)
        preserved = {key: value for key, value in self.meta.items() if key not in inspected}
        self.meta = {**preserved, **inspected}

    def save(self, dataset_obj: Dataset | DatasetDict) -> None:
        if not isinstance(dataset_obj, (Dataset, DatasetDict)):
            raise TypeError("DatasetArtifact.save() only accepts Dataset or DatasetDict")
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        dataset_obj.save_to_disk(str(self.path))
        inspected = self._inspect_dataset(dataset_obj)
        preserved = {key: value for key, value in self.meta.items() if key not in inspected}
        self.meta = {**preserved, **inspected}
        self.save_meta()

    def load(self) -> Dataset | DatasetDict:
        if not self.path.exists():
            self.materialize()
        if not self.path.is_dir():
            raise ValueError(f"DatasetArtifact path does not exist or is not a directory: {self.path}")
        try:
            dataset_obj = load_from_disk(str(self.path))
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise ValueError(f"Invalid Hugging Face datasets directory: {self.path}") from exc
        if not isinstance(dataset_obj, (Dataset, DatasetDict)):
            raise ValueError(f"Unsupported Hugging Face datasets object at {self.path}")
        return dataset_obj

    def materialize(self) -> None:
        if self.path.exists() and self.path.is_dir():
            try:
                self.refresh_metadata_from_disk()
                return
            except Exception:
                shutil.rmtree(self.path)

        if self.source is None:
            raise ValueError(
                "DatasetArtifact requires either an existing save_to_disk path or a source string"
            )

        dataset_obj = _load_dataset_source(self.source)
        working_dataset, selected_split, available_splits = select_working_dataset(dataset_obj)
        self.meta["source"] = str(self.source)
        self.meta["source_selected_split"] = selected_split
        self.meta["source_available_splits"] = available_splits
        self.save(working_dataset)

    def summary(self) -> dict[str, Any]:
        if not self.meta:
            self.refresh_metadata_from_disk()
        columns = self.meta.get("columns")
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.type,
            "hf_object_type": self.meta.get("hf_object_type"),
            "splits": self.meta.get("splits", []),
            "num_rows": self.meta.get("num_rows"),
            "columns": columns,
            "features": self.meta.get("features"),
            "fingerprint": self.meta.get("fingerprint"),
            "format": self.meta.get("format"),
            "dataset_format": self.meta.get("dataset_format"),
            "source": self.meta.get("source"),
        }


def _load_dataset_source(source: str) -> Dataset | DatasetDict:
    source_path = Path(source).expanduser()
    if source_path.exists():
        try:
            dataset_obj = load_from_disk(str(source_path))
        except Exception:
            dataset_obj = _load_dataset_from_local_source(source_path)
        if not isinstance(dataset_obj, (Dataset, DatasetDict)):
            raise ValueError(f"Unsupported dataset object loaded from local source: {source}")
        return dataset_obj

    try:
        dataset_obj = load_dataset(source)
    except Exception as exc:
        raise ValueError(f"Failed to load dataset source '{source}' from Hugging Face") from exc
    if not isinstance(dataset_obj, (Dataset, DatasetDict)):
        raise ValueError(f"Unsupported dataset object loaded from Hugging Face source: {source}")
    return dataset_obj


def _infer_split_data_files(files: list[str]) -> dict[str, list[str]] | None:
    split_aliases = {
        "train": "train",
        "test": "test",
        "validation": "validation",
        "valid": "validation",
        "eval": "eval",
        "dev": "dev",
    }
    grouped: dict[str, list[str]] = {}
    for file_name in files:
        stem = Path(file_name).stem.lower()
        matched_split = None
        for candidate, normalized in split_aliases.items():
            if stem == candidate or stem.startswith(f"{candidate}-") or stem.startswith(f"{candidate}_"):
                matched_split = normalized
                break
        if matched_split is None:
            return None
        grouped.setdefault(matched_split, []).append(file_name)
    return grouped or None


def _load_dataset_from_local_source(source_path: Path) -> Dataset | DatasetDict:
    if source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(source_path))
        if suffix == ".csv":
            return load_dataset("csv", data_files=str(source_path))
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(source_path))
        raise ValueError(f"Unsupported local dataset file format: {source_path}")

    if source_path.is_dir():
        suffix_to_builder = {
            ".json": "json",
            ".jsonl": "json",
            ".csv": "csv",
            ".parquet": "parquet",
        }
        files_by_builder: dict[str, list[str]] = {}
        for suffix, builder in suffix_to_builder.items():
            matched = sorted(str(path) for path in source_path.rglob(f"*{suffix}"))
            if matched:
                files_by_builder.setdefault(builder, []).extend(matched)
        if len(files_by_builder) == 1:
            builder, files = next(iter(files_by_builder.items()))
            split_files = _infer_split_data_files(files)
            return load_dataset(builder, data_files=split_files or files)
        try:
            return load_dataset(str(source_path))
        except Exception as exc:
            raise ValueError(f"Unsupported local dataset directory: {source_path}") from exc

    raise ValueError(f"Local dataset source does not exist: {source_path}")


@dataclass(slots=True)
class ModelArtifact(Artifact):
    def __post_init__(self) -> None:
        self.type = "model"
        Artifact.__post_init__(self)

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.type,
            "path": str(self.path),
            "mode": self.meta.get("mode"),
            "source_dataset_id": self.meta.get("source_dataset_id"),
            "base_model": self.meta.get("base_model"),
            "template": self.meta.get("template"),
            "finetuning_type": self.meta.get("finetuning_type"),
            "train_split": self.meta.get("train_split"),
            "eval_split": self.meta.get("eval_split"),
            "selected_rows": self.meta.get("selected_rows"),
            "log_path": self.meta.get("log_path"),
        }


@dataclass(slots=True)
class EvalArtifact(Artifact):
    def __post_init__(self) -> None:
        self.type = "eval"
        Artifact.__post_init__(self)

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.type,
            "path": str(self.path),
            "score": self.meta.get("score"),
            "source_model_id": self.meta.get("source_model_id"),
        }


@dataclass(slots=True)
class ResultArtifact(Artifact):
    def __post_init__(self) -> None:
        self.type = "result"
        Artifact.__post_init__(self)

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.type,
            "path": str(self.path),
            "keys": sorted(self.meta.keys()),
        }


@dataclass(slots=True)
class ExperimentArtifact(Artifact):
    def __post_init__(self) -> None:
        self.type = "experiment"
        Artifact.__post_init__(self)

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.type,
            "path": str(self.path),
            "status": self.meta.get("status"),
            "score": self.meta.get("score"),
            "steps": self.meta.get("num_steps"),
        }
