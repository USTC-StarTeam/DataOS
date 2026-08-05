"""The backend <-> launcher contract types.

A ``Backend`` turns a ``TrainSpec`` (see ``spec.py``) into a ``JobSpec``; a ``Launcher``
runs the ``JobSpec`` and returns a ``Handle``; ``Backend.collect`` parses outputs into a
``TrainResult``. These types are the wire format of the CLI (JSON in / JSON out), so they
carry no heavy deps and round-trip cleanly.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .io import make_id


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class Status(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"

    def is_terminal(self) -> bool:
        return self in (Status.COMPLETED, Status.FAILED, Status.TIMEOUT)


class Resources(_Model):
    nodes: int = 1
    gpus: int = 1  # PER NODE -> gres gpu:{gpus}; total devices = nodes * gpus
    cpus: int = 16
    time_h: float = 10.0
    mem_gb: int | None = None


class RayPlan(_Model):
    """How to bootstrap a Ray cluster for a ``launch.kind == 'ray'`` job (verl)."""

    head_gpus: int = 0
    start_head_cmd: list[str] | None = None
    worker_join_cmd: list[str] | None = None
    address_env: str = "RAY_ADDRESS"


class LaunchPlan(_Model):
    """How the program ``command`` is fanned into processes.

    The launcher composes the actual exec line from ``kind`` — the only place launch
    topology is interpreted — so launchers stay backend-agnostic.
    """

    kind: Literal["direct", "torchrun", "deepspeed", "ray"] = "direct"
    nproc_per_node: int | None = None  # defaults to resources.gpus
    nnodes: int | None = None  # defaults to resources.nodes
    rdzv: Literal["c10d", "static"] = "c10d"
    extra: list[str] = Field(default_factory=list)
    ray: RayPlan | None = None


class SlurmExtras(_Model):
    """Slurm placement, field names taken verbatim from ``run_config.to_cluster()``."""

    partition: str | None = None
    qos: str | None = None
    gres: str | None = None  # default derived as f"gpu:{resources.gpus}"
    nodelist: str | None = None
    cpus_per_task: int | None = None
    account: str | None = None
    extra_sbatch: list[str] = Field(default_factory=list)


class JobSpec(_Model):
    """A backend's rendered, launcher-agnostic job."""

    spec_id: str = Field(default_factory=lambda: make_id("job"))
    backend: str
    job_name: str
    command: list[str]  # argv of the user program, never a shell string
    env: dict[str, str] = Field(default_factory=dict)  # non-secret config only
    run_dir: str  # absolute; cwd AND where logs/handle/result are written
    resources: Resources = Field(default_factory=Resources)
    launch: LaunchPlan = Field(default_factory=LaunchPlan)
    conda_sh: str | None = None
    conda_env: str | None = None
    pythonpath: list[str] = Field(default_factory=list)
    pre_cmds: list[str] = Field(default_factory=list)
    # files the launcher materializes into run_dir before launch: rel_path -> contents
    assets: dict[str, str] = Field(default_factory=dict)
    slurm: SlurmExtras = Field(default_factory=SlurmExtras)


class Handle(_Model):
    """A live or finished job, persisted to ``<run_dir>/handle.json``."""

    launcher: Literal["slurm", "local"]
    spec_id: str
    run_dir: str
    job_name: str
    submitted_at: str
    # slurm
    job_id: str | None = None
    log_path: str | None = None
    # local
    pid: int | None = None
    hostname: str | None = None
    stdout_path: str | None = None
    exit_code: int | None = None


class TrainResult(_Model):
    """Canonical training outcome, produced by ``Backend.collect``."""

    spec_id: str
    status: Status
    model_path: str | None = None  # <run_dir>/final_model
    base_model: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    rows: int | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    log_tail: str | None = None
    # a ready-to-merge {stages.train, model, dataset} subtree matching DataOS report.json
    report_fragment: dict[str, Any] = Field(default_factory=dict)


class Capabilities(_Model):
    name: str
    methods: list[str]
    finetuning_types: list[str] = Field(default_factory=list)
    supports_multi_node: bool = False
    launch_kinds: list[str] = Field(default_factory=list)
    online_hook_points: list[str] = Field(default_factory=list)
    produces_format: str = "hf"
    needs_separate_env: bool = False
    env_domains: dict[str, str] = Field(
        default_factory=dict,
        description="method -> env.backends key, for methods whose dependency domain differs "
        "from the backend default; methods not listed use the backend name as their domain",
    )
    notes: str = ""
