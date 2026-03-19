"""Dataset version and lineage models for DataOS."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(value)


@dataclass(slots=True)
class LineageDependency:
    """Reference to an upstream dataset version."""

    dataset_id: str
    version: str
    storage_uri: str | None = None
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not self.version:
            raise ValueError("version must not be empty")
        self.metadata = _copy_mapping(self.metadata)

    @property
    def ref(self) -> str:
        return f"{self.dataset_id}:{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineageDependency":
        return cls(
            dataset_id=str(data["dataset_id"]),
            version=str(data["version"]),
            storage_uri=data.get("storage_uri"),
            role=data.get("role"),
            metadata=_copy_mapping(data.get("metadata")),
        )


@dataclass(slots=True)
class LineageRecord:
    """One lineage event that produced the dataset version."""

    upstreams: list[LineageDependency] = field(default_factory=list)
    operator_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None
    event_type: str = "transform"
    config: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.upstreams = [
            dependency
            if isinstance(dependency, LineageDependency)
            else LineageDependency.from_dict(dependency)
            for dependency in self.upstreams
        ]
        self.config = _copy_mapping(self.config)
        self.artifacts = _copy_mapping(self.artifacts)
        self.metadata = _copy_mapping(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineageRecord":
        return cls(
            upstreams=[
                dependency
                if isinstance(dependency, LineageDependency)
                else LineageDependency.from_dict(dependency)
                for dependency in data.get("upstreams", [])
            ],
            operator_id=data.get("operator_id"),
            run_id=data.get("run_id"),
            step_id=data.get("step_id"),
            event_type=str(data.get("event_type", "transform")),
            config=_copy_mapping(data.get("config")),
            artifacts=_copy_mapping(data.get("artifacts")),
            metadata=_copy_mapping(data.get("metadata")),
        )


@dataclass(slots=True)
class DatasetVersion:
    """Versioned dataset metadata with storage-friendly lineage."""

    dataset_id: str
    version: str
    source: dict[str, Any]
    schema: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    lineage: list[LineageRecord] = field(default_factory=list)
    storage_uri: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    created_by_run: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not self.version:
            raise ValueError("version must not be empty")
        self.source = _copy_mapping(self.source)
        self.schema = _copy_mapping(self.schema)
        self.statistics = _copy_mapping(self.statistics)
        self.tags = list(dict.fromkeys(self.tags))
        self.lineage = [
            record if isinstance(record, LineageRecord) else LineageRecord.from_dict(record)
            for record in self.lineage
        ]

    @property
    def ref(self) -> str:
        return f"{self.dataset_id}:{self.version}"

    @property
    def upstream_dependencies(self) -> list[LineageDependency]:
        dependencies: dict[str, LineageDependency] = {}
        for record in self.lineage:
            for dependency in record.upstreams:
                dependencies[dependency.ref] = dependency
        return list(dependencies.values())

    def add_lineage(self, record: LineageRecord | dict[str, Any]) -> None:
        self.lineage.append(
            record if isinstance(record, LineageRecord) else LineageRecord.from_dict(record)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metadata_dict(self) -> dict[str, Any]:
        metadata = self.to_dict()
        metadata.pop("lineage", None)
        return metadata

    def lineage_dict(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.lineage]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetVersion":
        return cls(
            dataset_id=str(data["dataset_id"]),
            version=str(data["version"]),
            source=_copy_mapping(data.get("source")),
            schema=_copy_mapping(data.get("schema")),
            statistics=_copy_mapping(data.get("statistics")),
            tags=list(data.get("tags", [])),
            lineage=[
                record if isinstance(record, LineageRecord) else LineageRecord.from_dict(record)
                for record in data.get("lineage", [])
            ],
            storage_uri=data.get("storage_uri"),
            created_at=str(data.get("created_at", _utc_now_iso())),
            created_by_run=data.get("created_by_run"),
        )
