"""Filesystem-backed dataset version storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from dataos.core import DatasetVersion


class DatasetStore:
    """Persist versioned datasets under ``<root>/<dataset_id>/<version>/``."""

    SUPPORTED_FORMATS = frozenset({"jsonl", "parquet"})

    def __init__(self, root_dir: str | Path = "artifacts/datasets") -> None:
        self.root_dir = Path(root_dir)

    def dataset_dir(self, dataset_id: str, version: str) -> Path:
        return self.root_dir / dataset_id / version

    def save(
        self,
        dataset: DatasetVersion,
        rows: Iterable[Mapping[str, Any]],
        *,
        format: str | None = None,
        overwrite: bool = False,
    ) -> DatasetVersion:
        storage_format = self._resolve_format(dataset, format)
        dataset_dir = self.dataset_dir(dataset.dataset_id, dataset.version)
        data_path = dataset_dir / self._data_filename(storage_format)
        metadata_path = dataset_dir / "metadata.json"
        lineage_path = dataset_dir / "lineage.json"

        self._ensure_writable_paths(
            [data_path, metadata_path, lineage_path],
            overwrite=overwrite,
        )

        dataset_dir.mkdir(parents=True, exist_ok=True)
        stored_dataset = DatasetVersion.from_dict(dataset.to_dict())
        stored_dataset.storage_uri = dataset_dir.as_posix()
        stored_dataset.source["format"] = storage_format

        self._write_rows(data_path, rows, storage_format)
        self._write_json(
            metadata_path,
            {
                **stored_dataset.metadata_dict(),
                "storage_format": storage_format,
                "data_filename": data_path.name,
            },
        )
        self._write_json(lineage_path, stored_dataset.lineage_dict())
        return stored_dataset

    def save_dataset(
        self,
        dataset: DatasetVersion,
        rows: Iterable[Mapping[str, Any]],
        *,
        format: str | None = None,
        overwrite: bool = False,
    ) -> DatasetVersion:
        return self.save(dataset, rows, format=format, overwrite=overwrite)

    def load(self, dataset_id: str, version: str) -> tuple[DatasetVersion, list[dict[str, Any]]]:
        dataset = self.load_dataset(dataset_id, version)
        rows = self.load_rows(dataset_id, version)
        return dataset, rows

    def load_dataset(self, dataset_id: str, version: str) -> DatasetVersion:
        dataset_dir = self.dataset_dir(dataset_id, version)
        metadata_path = dataset_dir / "metadata.json"
        lineage_path = dataset_dir / "lineage.json"
        metadata = self._read_json(metadata_path)
        lineage = self._read_json(lineage_path) if lineage_path.exists() else []

        storage_format = metadata.pop("storage_format", None)
        metadata.pop("data_filename", None)
        payload = {**metadata, "lineage": lineage}
        dataset = DatasetVersion.from_dict(payload)
        if storage_format is not None:
            dataset.source["format"] = str(storage_format).lower()
        if dataset.storage_uri is None:
            dataset.storage_uri = dataset_dir.as_posix()
        return dataset

    def load_rows(self, dataset_id: str, version: str) -> list[dict[str, Any]]:
        dataset_dir = self.dataset_dir(dataset_id, version)
        metadata = self._read_json(dataset_dir / "metadata.json")
        storage_format = str(metadata.get("storage_format", self._infer_format(dataset_dir))).lower()
        data_filename = str(metadata.get("data_filename", self._data_filename(storage_format)))
        data_path = dataset_dir / data_filename
        return self._read_rows(data_path, storage_format)

    def save_parquet(self, data_path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
        raise NotImplementedError("parquet dataset storage is not implemented yet")

    def load_parquet(self, data_path: str | Path) -> list[dict[str, Any]]:
        raise NotImplementedError("parquet dataset storage is not implemented yet")

    def _resolve_format(self, dataset: DatasetVersion, format: str | None) -> str:
        storage_format = str(format or dataset.source.get("format") or "jsonl").lower()
        if storage_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"unsupported dataset format: {storage_format}")
        return storage_format

    def _data_filename(self, storage_format: str) -> str:
        if storage_format == "jsonl":
            return "data.jsonl"
        if storage_format == "parquet":
            return "data.parquet"
        raise ValueError(f"unsupported dataset format: {storage_format}")

    def _infer_format(self, dataset_dir: Path) -> str:
        if (dataset_dir / "data.jsonl").exists():
            return "jsonl"
        if (dataset_dir / "data.parquet").exists():
            return "parquet"
        raise FileNotFoundError(f"no dataset payload found under {dataset_dir}")

    def _write_rows(
        self,
        data_path: Path,
        rows: Iterable[Mapping[str, Any]],
        storage_format: str,
    ) -> None:
        if storage_format == "jsonl":
            self._write_jsonl(data_path, rows)
            return
        if storage_format == "parquet":
            self.save_parquet(data_path, rows)
            return
        raise ValueError(f"unsupported dataset format: {storage_format}")

    def _read_rows(self, data_path: Path, storage_format: str) -> list[dict[str, Any]]:
        if storage_format == "jsonl":
            return self._read_jsonl(data_path)
        if storage_format == "parquet":
            return self.load_parquet(data_path)
        raise ValueError(f"unsupported dataset format: {storage_format}")

    def _write_jsonl(self, data_path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        temp_path = data_path.with_suffix(f"{data_path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError("dataset rows must be mappings")
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        temp_path.replace(data_path)

    def _read_jsonl(self, data_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"jsonl row at line {line_number} in {data_path} must decode to an object"
                    )
                rows.append(value)
        return rows

    def _write_json(self, path: Path, payload: Any) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(path)

    def _read_json(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _ensure_writable_paths(self, paths: Iterable[Path], *, overwrite: bool) -> None:
        if overwrite:
            return
        existing = [path for path in paths if path.exists()]
        if existing:
            joined = ", ".join(path.as_posix() for path in existing)
            raise FileExistsError(f"dataset version already exists: {joined}")
