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
from typing import Any, Literal

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
    """What the job needs, in units the job understands — not in a scheduler's units.

    ``gpus`` is per node; total devices are ``nodes * gpus``. **Zero means this job does not use a
    GPU**, which is a statement about the program, not a request to the scheduler — and the two
    come apart. On PTB's partition ``gres_required: "gpu:1"`` means even a pure-CPU data build is
    allocated a card; on the guarded QOS, holding that card idle is precisely what gets the job
    scancelled. Saying "I don't use a GPU" is what lets the launcher do both right things: ask for
    the card anyway, and install the guard placeholder. A job that silently passed ``gpus=1`` to
    mean "give me the mandatory card" could not be told apart from one that trains on it.

    ``mem_gb`` is likewise a need, not an ``--mem`` line. Where memory is granted per card, the
    launcher converts: wanting more than one card's worth means wanting more cards.
    """

    nodes: int = 1
    gpus: int = 1
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


class Job(Model):
    """A stage's rendered job: what to run, and what it needs to run it.

    Launcher-agnostic, and now literally so: there is no scheduler word in this type. It used to
    carry a ``slurm: SlurmExtras`` block, which meant a job destined for a bare-metal launcher
    still described a partition and a QOS, and — worse — that block took precedence over the
    launcher's own configuration (``s.qos or self.defaults.get("qos")``), so a stage's guess
    overrode the deployment's answer. Placement belongs to the launcher, which is built from
    config; this says only what to run and what it needs.
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


class Handle(Model):
    """A live or finished job, persisted to ``<run_dir>/handle.json``.

    The envelope is the core's; the contents are the launcher's. This used to be the union of
    both launchers' fields — ``job_id``/``log_path`` for Slurm, ``pid``/``hostname``/
    ``stdout_path``/``exit_code`` for local — which pushed the choice onto every caller:
    ``handle.job_id or str(handle.pid)`` appeared in dataos_eval, and
    ``handle.log_path or handle.stdout_path`` in dataos_train. Two universal questions were being
    asked twice each, once per launcher.

    So two universal answers stay on the envelope — every scheduler names its jobs, and every
    launcher writes their output somewhere — and everything else moves into ``native``, which the
    launcher writes and reads and nothing else interprets.

    ``launcher`` is a plain string, not a Literal of the two built-ins. ``registry`` loads
    launchers from the ``dataos_launch.launchers`` entry-point group, so a third-party launcher is
    invited by one file and was made unrepresentable by another.
    """

    launcher: str
    spec_id: str
    run_dir: str
    job_name: str
    submitted_at: str
    job_id: str  # what this launcher calls it
    log_path: str | None = None  # where its output is
    native: dict[str, Any] = Field(default_factory=dict)  # the launcher's own bookkeeping
