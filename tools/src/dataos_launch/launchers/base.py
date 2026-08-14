"""The Launcher base: what every launcher shares, and the two things each one supplies.

A launcher answers WHERE a ``Job`` runs. Submitting has a fixed shape — materialize the assets,
render a script, hand it to whoever starts things here, record what came back — and each of the
two built-ins used to carry its own copy of it. Which is how ``handle.json`` came to be written
twice with two different field sets, and how ``logs()`` came to be implemented twice over two
differently-named path fields. Here the shape is written once and a launcher supplies only what
genuinely differs: :meth:`Launcher.render` and :meth:`Launcher.dispatch`.

This is a base class rather than the ``Protocol`` it was until 2026-08-14. The Protocol's stated
reason — "so a launcher can come from anywhere without importing this one at definition time" —
never held: any implementation has to import ``Job``, ``Handle`` and ``Status`` from this package
to be typed at all, so it already imports this package. The cost was real, though: a structural
contract cannot carry an implementation, so ``submit`` could not be shared, and the duplication
above is what filled the gap.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from ..io import atomic_write_json, atomic_write_text, utc_now_iso
from ..types import Handle, Job, Status


class LaunchError(RuntimeError):
    pass


def materialize(job: Job) -> Path:
    """Create ``run_dir`` (+ ``logs/``) and write all declared assets. Returns run_dir."""
    run_dir = Path(job.run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    for rel, contents in job.assets.items():
        atomic_write_text(run_dir / rel, contents)
    return run_dir


#: Env every training job gets unless its own ``env`` overrides the key.
#: ``expandable_segments`` lets the caching allocator grow an existing segment instead of
#: failing while GiB sit reserved-but-unallocated. A full-SFT step allocates one large,
#: short-lived fp32 logits block per micro-batch (tokens x vocab x 4B — 2.3 GiB at 4096
#: tokens on a 150k-vocab model), which fragments the pool badly; 2026-08-10's 4b2 v5 batch
#: lost 32 jobs to OOMs that reported 3.46 GiB reserved but unallocated.
_ENV_DEFAULTS = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def build_preamble(job: Job) -> list[str]:
    """Bash setup lines before the launch line (conda + PYTHONPATH + env + cd)."""
    lines = ["set -euo pipefail"]
    if job.conda_sh and job.conda_env:
        lines.append(
            f"source {shlex.quote(job.conda_sh)} && conda activate {shlex.quote(job.conda_env)}"
        )
    elif job.conda_env and job.conda_env.startswith("/"):
        # standard venv (uv/python -m venv): no conda needed — activate by path
        lines.append(f"source {shlex.quote(job.conda_env)}/bin/activate")
    if job.pythonpath:
        joined = ":".join(job.pythonpath)
        lines.append(f"export PYTHONPATH={shlex.quote(joined)}${{PYTHONPATH:+:$PYTHONPATH}}")
    for key, val in {**_ENV_DEFAULTS, **job.env}.items():
        lines.append(f"export {key}={shlex.quote(val)}")
    lines.extend(job.pre_cmds)
    lines.append(f"cd {shlex.quote(job.run_dir)}")
    return lines


def compose_launch_line(job: Job) -> str:
    """Render the exec line from ``launch.kind``. ``command`` is the user program's argv."""
    plan = job.launch
    quoted = " ".join(shlex.quote(c) for c in job.command)
    nproc = plan.nproc_per_node or job.resources.gpus
    nnodes = plan.nnodes or job.resources.nodes

    if plan.kind == "direct":
        return quoted
    if plan.kind == "torchrun":
        parts = [
            "torchrun",
            "--nnodes",
            str(nnodes),
            "--nproc-per-node",
            str(nproc),
            "--rdzv-backend",
            plan.rdzv,
            *plan.extra,
            "--",
            *job.command,
        ]
        return " ".join(shlex.quote(p) if " " in p else p for p in parts)
    if plan.kind == "deepspeed":
        parts = ["deepspeed", "--num_gpus", str(nproc), *plan.extra, *job.command]
        return " ".join(shlex.quote(p) if " " in p else p for p in parts)
    raise LaunchError(f"unknown launch kind: {plan.kind!r}")


class Launcher(ABC):
    """Where a Job runs.

    Built from configuration, never from a Job — see
    :func:`dataos_launch.placement.launcher_from_config`. Placement (partition, QOS, node pool)
    is a property of the deployment plus the run's choice; a stage that could also state it
    would be a third opinion, and it silently outranked the other two for months.
    """

    #: registry key, and the value written to ``Handle.launcher``
    name: ClassVar[str]
    #: appended to ``job_name`` to name the generated script
    script_suffix: ClassVar[str] = ".sh"

    def submit(self, job: Job) -> Handle:
        """Materialize, render, dispatch, record. The one submission path."""
        run_dir = materialize(job)
        script = run_dir / f"{job.job_name}{self.script_suffix}"
        atomic_write_text(script, self.render(job))
        handle = self.dispatch(job, script)
        atomic_write_json(run_dir / "handle.json", handle.model_dump())
        return handle

    def logs(self, handle: Handle, *, tail: int | None = None) -> str:
        """The job's output. One implementation, now that every Handle has a ``log_path``."""
        if not handle.log_path:
            return ""
        path = Path(handle.log_path)
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-tail:]) if tail is not None else text

    def new_handle(
        self, job: Job, *, job_id: str, log_path: str | None = None, **native: Any
    ) -> Handle:
        """Fill the envelope; ``native`` is whatever only this launcher needs in order to poll."""
        return Handle(
            launcher=self.name,
            spec_id=job.spec_id,
            run_dir=job.run_dir,
            job_name=job.job_name,
            submitted_at=utc_now_iso(),
            job_id=job_id,
            log_path=log_path,
            native=native,
        )

    @abstractmethod
    def render(self, job: Job) -> str:
        """The script text to hand to whatever starts jobs here."""

    @abstractmethod
    def dispatch(self, job: Job, script: Path) -> Handle:
        """Start the written script; return the Handle (built via :meth:`new_handle`)."""

    @abstractmethod
    def status(self, handle: Handle) -> Status: ...

    @abstractmethod
    def cancel(self, handle: Handle) -> None: ...
