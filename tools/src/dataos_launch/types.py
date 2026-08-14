"""What a stage hands the launcher, and what it gets back.

Three nouns, one per layer: a **Spec** says what a stage intends, a **Job** says what to run and
what it needs, a **Handle** says who took it and what they call it. The middle one used to be
``JobSpec``, which read as a kind of Spec and is not — it is what a Spec becomes once the
deployment is resolved into it.

These types are the CLI's wire format (JSON in, JSON out, written to disk and read back rounds
later), so they carry no heavy deps and round-trip cleanly.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from .io import make_id
from .model import Model


class Status(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"

    def is_terminal(self) -> bool:
        return self in (Status.COMPLETED, Status.FAILED, Status.TIMEOUT)


class Resources(Model):
    nodes: int = 1
    gpus: int = 1  # PER NODE -> gres gpu:{gpus}; total devices = nodes * gpus
    cpus: int = 16
    time_h: float = 10.0
    mem_gb: int | None = None


class LaunchPlan(Model):
    """How the program ``command`` is fanned into processes.

    The launcher composes the actual exec line from ``kind`` — the only place launch
    topology is interpreted — so launchers stay backend-agnostic.
    """

    kind: Literal["direct", "torchrun", "deepspeed"] = "direct"
    nproc_per_node: int | None = None  # defaults to resources.gpus
    nnodes: int | None = None  # defaults to resources.nodes
    rdzv: Literal["c10d", "static"] = "c10d"
    extra: list[str] = Field(default_factory=list)


class SlurmExtras(Model):
    """Slurm placement, field names taken verbatim from ``run_config.to_cluster()``."""

    partition: str | None = None
    qos: str | None = None
    gres: str | None = None  # default derived as f"gpu:{resources.gpus}"
    nodelist: str | None = None
    cpus_per_task: int | None = None
    account: str | None = None
    extra_sbatch: list[str] = Field(default_factory=list)


class Job(Model):
    """A stage's rendered job: what to run, and what it needs to run it.

    Launcher-agnostic, which the ``slurm: SlurmExtras`` field below currently contradicts — that
    is the next commit.
    """

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


class Handle(Model):
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


