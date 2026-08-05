"""可直接接入算子的 dataset 容器和加载入口。

canonical 存储是 list-of-dicts：record 是任意异构 dict，能被无损保留——类型、缺失
字段都按原样 round-trip，算子和字段路径工具完全无感。Hugging Face
``datasets.Dataset``（Arrow，列式）只作为**导出/落盘**形态，在 ``hf_dataset`` /
``save_to_disk`` / ``push_to_hf`` 时按需转换；转换时 free-form 的 ``metadata`` 字段
以 JSON 字符串列保存，从而规避 Arrow 对 ragged 嵌套 struct 的统一限制。

之所以不把 Arrow 当内存存储：Arrow 每列要求单一类型、且每行都有该列，会把首条
记录缺失的字段整列丢弃、并对异构类型（如同一字段既有 ``"2.5"`` 又有 ``1``）报错——
这与"record 是任意 dict"的核心契约冲突。

jsonl / json / csv 只作为导入、训练导出和人/agent 查看的边界格式；
``save_to_disk`` / ``from_disk`` 才是全保真、可内存映射的工作格式。``datasets`` 仅在
真正需要 Arrow 时才 import，因此非强制依赖，也便于测试 mock。
"""

from __future__ import annotations

import contextlib
import csv
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, overload

from dataos_operator.core.contracts import Record, get_field

DatasetT = TypeVar("DatasetT", bound="Dataset")
_MISSING = object()
_METADATA_COLUMN = "metadata"
_METADATA_SIDECAR = "dataos_metadata.json"


class Dataset(Sequence[Record]):
    """可以直接传给 DataOS 算子的标准 dataset 对象。"""

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        table: Any | None = None,
    ) -> None:
        self._metadata: dict[str, Any] = dict(metadata or {})
        if table is not None:
            self._records: list[dict[str, Any]] = [_decode_row(row) for row in table]
        else:
            self._records = [dict(record) for record in (records or [])]

    @overload
    def __getitem__(self, index: int) -> Record: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Record]: ...

    def __getitem__(self, index: int | slice) -> Record | Sequence[Record]:
        if isinstance(index, slice):
            return [dict(record) for record in self._records[index]]
        return dict(self._records[index])

    def __iter__(self) -> Iterator[Record]:
        for record in self._records:
            yield dict(record)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def metadata(self) -> Mapping[str, Any]:
        """返回 dataset 来源等元信息。"""

        return MappingProxyType(self._metadata)

    @property
    def hf_dataset(self) -> Any:
        """按需把 records 构造成 Hugging Face ``datasets.Dataset``（Arrow 表）。

        canonical 存储是 list-of-dicts；只有导出到 HF / Arrow 时才在此处转换，
        ``metadata`` 列会 JSON 编码以规避 ragged struct。需要安装 ``datasets``。
        """

        return _build_table(self._records)

    def to_records(self) -> list[Record]:
        """返回浅拷贝后的 records 列表。"""

        return [dict(record) for record in self._records]

    def save_local(
        self,
        path: str | Path,
        *,
        file_format: str = "auto",
        overwrite: bool = True,
    ) -> Path:
        """把 dataset 保存到本地 JSON、JSONL 或 CSV 文件。"""

        file_path = Path(path)
        output_format = _resolve_output_format(file_path, file_format)
        _ensure_writable_path(file_path, overwrite=overwrite)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        records = self.to_records()
        if output_format == "jsonl":
            _write_jsonl(file_path, records)
        elif output_format == "json":
            _write_json(file_path, records)
        elif output_format == "csv":
            _write_csv(file_path, records)
        else:
            raise ValueError(f"Unsupported dataset output format: {output_format!r}")
        return file_path

    def save_to_disk(self, path: str | Path) -> Path:
        """把 dataset 以 Hugging Face Arrow 格式保存到本地目录（全保真、可内存映射）。

        dataset 级元信息（``source`` / ``history`` 等）写入目录内的 sidecar 文件，
        从而修复 jsonl 导出会丢失 dataset 级 metadata 的问题。
        """

        target = Path(path)
        self.hf_dataset.save_to_disk(str(target))
        if self._metadata:
            (target / _METADATA_SIDECAR).write_text(
                json.dumps(self._metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return target

    @classmethod
    def from_disk(cls: type[DatasetT], path: str | Path) -> DatasetT:
        """从 ``save_to_disk`` 产生的 Hugging Face Arrow 目录构造 dataset。"""

        import datasets as hfds

        target = Path(path)
        table = hfds.load_from_disk(str(target))
        metadata: dict[str, Any] = {}
        sidecar = target / _METADATA_SIDECAR
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        return cls(table=table, metadata=metadata)

    def push_to_hf(
        self,
        repo_id: str,
        *,
        split: str = "train",
        config_name: str = "default",
        token: str | None = None,
        private: bool | None = None,
        commit_message: str | None = None,
        create_pr: bool = False,
        **push_kwargs: Any,
    ) -> Any:
        """把 dataset 上传到 Hugging Face Hub。"""

        import datasets as hfds

        hf_dataset = hfds.Dataset.from_list(self.to_records())
        return hf_dataset.push_to_hub(
            repo_id,
            split=split,
            config_name=config_name,
            token=token,
            private=private,
            commit_message=commit_message,
            create_pr=create_pr,
            **push_kwargs,
        )

    @classmethod
    def from_records(
        cls: type[DatasetT],
        records: Iterable[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetT:
        """从内存中的 records 构造 dataset。"""

        return cls(records=records, metadata=metadata)

    @classmethod
    def from_source(
        cls: type[DatasetT],
        source: str | Path,
        *,
        source_type: str = "auto",
        **kwargs: Any,
    ) -> DatasetT:
        """根据本地路径或 `hf://` 路径构造 dataset。"""

        source_text = str(source)
        if source_type == "auto":
            if source_text.startswith("hf://"):
                return cls.from_hf(source_text.removeprefix("hf://"), **kwargs)
            return cls.from_local(source, **kwargs)
        if source_type == "hf":
            return cls.from_hf(source_text.removeprefix("hf://"), **kwargs)
        if source_type == "local":
            return cls.from_local(source, **kwargs)
        raise ValueError(f"Unsupported source_type: {source_type!r}")

    @classmethod
    def from_local(
        cls: type[DatasetT],
        path: str | Path,
        *,
        source_name: str | None = None,
        field_map: Mapping[str, str] | None = None,
        limit: int | None = None,
        id_prefix: str | None = None,
        target_regex: str | None = None,
        target_source_key: str = "answer",
        target_key: str = "target",
    ) -> DatasetT:
        """从本地 JSON、JSONL 或 CSV 文件构造 dataset。"""

        file_path = Path(path)
        raw_records = _read_local_records(file_path)
        resolved_source_name = source_name or file_path.stem
        records = _normalize_records(
            raw_records,
            field_map=field_map,
            limit=limit,
            id_prefix=id_prefix,
            target_regex=target_regex,
            target_source_key=target_source_key,
            target_key=target_key,
            record_source=resolved_source_name,
            record_metadata=None,
        )
        return cls(
            records=records,
            metadata={
                "source": resolved_source_name,
                "loader": "local",
                "path": str(file_path),
                "records": len(records),
            },
        )

    @classmethod
    def from_hf(
        cls: type[DatasetT],
        path: str,
        *,
        source_name: str | None = None,
        name: str | None = None,
        split: str = "train",
        field_map: Mapping[str, str] | None = None,
        limit: int | None = None,
        streaming: bool = False,
        id_prefix: str | None = None,
        target_regex: str | None = None,
        target_source_key: str = "answer",
        target_key: str = "target",
        **load_kwargs: Any,
    ) -> DatasetT:
        """从 Hugging Face Hub 构造 dataset。"""

        import datasets as hfds

        args: list[str] = [path]
        if name is not None:
            args.append(name)
        hf_dataset = hfds.load_dataset(*args, split=split, streaming=streaming, **load_kwargs)
        resolved_source_name = source_name or path.rsplit("/", 1)[-1]
        record_metadata = {
            "path": path,
            "split": split,
            "loader": "huggingface",
        }
        if name is not None:
            record_metadata["name"] = name

        records = _normalize_records(
            hf_dataset,
            field_map=field_map,
            limit=limit,
            id_prefix=id_prefix or _default_hf_id_prefix(path, split),
            target_regex=target_regex,
            target_source_key=target_source_key,
            target_key=target_key,
            record_source=resolved_source_name,
            record_metadata=record_metadata,
        )
        return cls(
            records=records,
            metadata={
                "source": resolved_source_name,
                **record_metadata,
                "records": len(records),
                "streaming": streaming,
            },
        )


def _build_table(records: list[Mapping[str, Any]]) -> Any:
    """把 dict records 构造成底层 Arrow 表，``metadata`` 列 JSON 编码为字符串。"""

    import datasets as hfds

    encoded: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        value = row.get(_METADATA_COLUMN)
        if isinstance(value, (Mapping, list)):
            row[_METADATA_COLUMN] = json.dumps(value, ensure_ascii=False, default=str)
        encoded.append(row)
    if not encoded:
        return hfds.Dataset.from_dict({})
    try:
        return hfds.Dataset.from_list(encoded)
    except Exception as exc:  # noqa: BLE001 - surface a clearer schema-conflict error
        raise ValueError(
            "Failed to build an Arrow dataset from records; this usually means a column has "
            f"conflicting types across records (Arrow requires a unifiable schema): {exc}"
        ) from exc


def _decode_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """把底层 Arrow row 转回普通 dict，并解码 ``metadata`` JSON 列。"""

    decoded = dict(row)
    value = decoded.get(_METADATA_COLUMN)
    if isinstance(value, str):
        with contextlib.suppress(json.JSONDecodeError):
            decoded[_METADATA_COLUMN] = json.loads(value)
    return decoded


def _read_local_records(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        return _read_json_records(path)
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported dataset file suffix: {suffix!r}")


def _resolve_output_format(path: Path, file_format: str) -> str:
    if file_format != "auto":
        requested_format = file_format.lower()
        if requested_format == "ndjson":
            return "jsonl"
        return requested_format
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Cannot infer dataset output format from suffix: {suffix!r}")


def _ensure_writable_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset file already exists: {path}")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        json.dumps(list(records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = _csv_fieldnames(records)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_cell_value(record.get(key)) for key in fieldnames})


def _csv_fieldnames(records: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def _csv_cell_value(value: Any) -> Any:
    if isinstance(value, Mapping | list):
        return json.dumps(value, ensure_ascii=False)
    return value


def _read_json_records(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [_ensure_mapping(item, path) for item in value]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return [_ensure_mapping(item, path) for item in value["records"]]
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        return [_ensure_mapping(item, path) for item in value["data"]]
    raise ValueError(f"JSON dataset must be a list or contain records/data list: {path}")


def _ensure_mapping(value: object, path: Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"Dataset item must be an object in {path}: {value!r}")


def _normalize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    field_map: Mapping[str, str] | None,
    limit: int | None,
    id_prefix: str | None,
    target_regex: str | None,
    target_source_key: str,
    target_key: str,
    record_source: str,
    record_metadata: Mapping[str, Any] | None,
) -> list[Record]:
    normalized_records: list[Record] = []
    for index, raw_record in enumerate(records):
        if limit is not None and index >= limit:
            break
        normalized_records.append(
            _normalize_record(
                raw_record,
                index=index,
                field_map=field_map,
                id_prefix=id_prefix,
                target_regex=target_regex,
                target_source_key=target_source_key,
                target_key=target_key,
                record_source=record_source,
                record_metadata=record_metadata,
            )
        )
    return normalized_records


def _normalize_record(
    raw_record: Mapping[str, Any],
    *,
    index: int,
    field_map: Mapping[str, str] | None,
    id_prefix: str | None,
    target_regex: str | None,
    target_source_key: str,
    target_key: str,
    record_source: str,
    record_metadata: Mapping[str, Any] | None,
) -> Record:
    if field_map is None:
        record = dict(raw_record)
    else:
        record = {
            target_field: _require_source_field(raw_record, source_field)
            for target_field, source_field in field_map.items()
        }

    if id_prefix is not None and "id" not in record:
        record["id"] = f"{id_prefix}-{index:05d}"

    if target_regex is not None:
        source_value = _require_source_field(record, target_source_key)
        match = re.search(target_regex, str(source_value), flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"Target regex did not match record at index {index}")
        record[target_key] = match.group(1)

    current_metadata = record.get("metadata")
    if not isinstance(current_metadata, Mapping):
        current_metadata = {}
    normalized_metadata: dict[str, Any] = {
        **current_metadata,
        "source": record_source,
    }
    if record_metadata is not None:
        normalized_metadata.update(record_metadata)
        normalized_metadata["index"] = index
    record["metadata"] = normalized_metadata

    return record


def _require_source_field(record: Mapping[str, Any], key: str) -> Any:
    value = get_field(record, key, _MISSING)
    if value is _MISSING:
        raise KeyError(f"Source field is missing: {key}")
    return value


def _default_hf_id_prefix(path: str, split: str) -> str:
    raw_prefix = f"{path}-{split}"
    return re.sub(r"[^a-zA-Z0-9]+", "-", raw_prefix).strip("-").lower()
