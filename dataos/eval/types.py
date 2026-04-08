from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalTaskConfig:
    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalDataConfig:
    split: str | None = None
    max_samples: int | None = None


@dataclass(slots=True)
class EvalAdapterConfig:
    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRequestConfig:
    type: str
    batch_size: int = 8
    max_new_tokens: int = 256
    do_sample: bool = False


@dataclass(slots=True)
class EvalArtifactPolicy:
    persist_requests: bool = False
    persist_responses: bool = False
    persist_samples: bool = True


@dataclass(slots=True)
class EvalConfig:
    provider: str
    task: EvalTaskConfig | None = None
    data: EvalDataConfig = field(default_factory=EvalDataConfig)
    adapter: EvalAdapterConfig | None = None
    request: EvalRequestConfig | None = None
    artifacts: EvalArtifactPolicy = field(default_factory=EvalArtifactPolicy)
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRequest:
    sample_id: str
    index: int
    request_type: str
    prompt: str
    stop: list[str] | None = None
    choices: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelResponse:
    sample_id: str
    index: int
    request_type: str
    text: str | None = None
    logprob: float | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalSampleResult:
    sample_id: str
    index: int
    prompt: str
    prediction: str | None
    reference: Any
    normalized_prediction: Any = None
    normalized_reference: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalResultBundle:
    metric_name: str
    score: float
    summary: dict[str, Any]
    samples: list[EvalSampleResult]


@dataclass(slots=True)
class EvalRunOutput:
    provider: str
    task_name: str | None
    task_version: str | None
    adapter_name: str | None
    request_type: str | None
    dataset_split: str | None
    summary: EvalResultBundle | None
    requests: list[EvalRequest] = field(default_factory=list)
    responses: list[ModelResponse] = field(default_factory=list)
    sample_results: list[EvalSampleResult] = field(default_factory=list)
