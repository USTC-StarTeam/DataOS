"""Slurm launcher: sbatch render + sacct poll.

Runs **locally on the cluster** (login node) — ssh/rsync stays in the DataOS deploy/sub-agent
layer, never inside the launcher.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..io import atomic_write_json, atomic_write_text, utc_now_iso
from ..types import Handle, JobSpec, Status
from .base import build_preamble, compose_launch_line, materialize

_FAIL = {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "PREEMPTED", "REVOKED"}
_TIMEOUT = {"TIMEOUT", "DEADLINE"}
_RUNNING = {"RUNNING", "COMPLETING", "RESIZING", "SUSPENDED"}


def map_sacct_state(raw: str) -> Status:
    """Map a sacct ``State`` field to a ``Status`` (handles leading spaces + reasons)."""
    token = (raw or "").strip()
    state = token.split()[0].upper() if token else ""
    if state == "COMPLETED":
        return Status.COMPLETED
    if state in _TIMEOUT:
        return Status.TIMEOUT
    if state in _FAIL or state.startswith("CANCELLED"):
        return Status.FAILED
    if state in _RUNNING:
        return Status.RUNNING
    return Status.PENDING  # PENDING/CONFIGURING/REQUEUED/empty/not-yet-registered


def _hms(time_h: float) -> str:
    total = max(1, int(round(time_h * 3600)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class SlurmLauncher:
    name = "slurm"

    def __init__(self, **defaults: object) -> None:
        # cluster-level defaults (partition/qos/nodelist) e.g. from run_config.to_cluster()
        self.defaults = defaults

    def render_sbatch(self, job: JobSpec) -> str:
        r = job.resources
        s = job.slurm
        partition = s.partition or self.defaults.get("partition")
        qos = s.qos or self.defaults.get("qos")
        gres = s.gres or f"gpu:{r.gpus}"
        cpus = s.cpus_per_task or r.cpus
        nodelist = s.nodelist or self.defaults.get("nodelist")

        lines = [
            "#!/bin/bash",
            f"#SBATCH -J {job.job_name}",
            "#SBATCH -o slurm-%j.out",
            "#SBATCH -e slurm-%j.out",
            f"#SBATCH --nodes={r.nodes}",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --gres={gres}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --time={_hms(r.time_h)}",
        ]
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if qos:
            lines.append(f"#SBATCH --qos={qos}")
        if nodelist:
            lines.append(f"#SBATCH --nodelist={nodelist}")
        if s.account:
            lines.append(f"#SBATCH --account={s.account}")
        if r.mem_gb:
            lines.append(f"#SBATCH --mem={r.mem_gb}G")
        for extra in s.extra_sbatch:
            lines.append(extra if extra.lstrip().startswith("#SBATCH") else f"#SBATCH {extra}")
        lines.append("")
        lines.extend(build_preamble(job))
        lines.append(compose_launch_line(job))
        return "\n".join(lines) + "\n"

    def submit(self, job: JobSpec) -> Handle:
        materialize(job)
        run_dir = Path(job.run_dir)
        script = run_dir / f"{job.job_name}.sbatch"
        atomic_write_text(script, self.render_sbatch(job))
        out = subprocess.run(
            ["sbatch", "--parsable", str(script)],
            cwd=job.run_dir,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            raise RuntimeError(f"sbatch failed: {out.stderr.strip() or out.stdout.strip()}")
        job_id = out.stdout.strip().split(";")[0].strip()
        if not re.fullmatch(r"\d+", job_id):
            raise RuntimeError(f"could not parse job id from sbatch output: {out.stdout!r}")
        handle = Handle(
            launcher="slurm",
            spec_id=job.spec_id,
            run_dir=job.run_dir,
            job_name=job.job_name,
            submitted_at=utc_now_iso(),
            job_id=job_id,
            log_path=str(run_dir / f"slurm-{job_id}.out"),
        )
        atomic_write_json(run_dir / "handle.json", handle.model_dump())
        return handle

    def status(self, handle: Handle) -> Status:
        out = subprocess.run(
            ["sacct", "-j", str(handle.job_id), "--format=State", "--noheader", "-n"],
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            return Status.PENDING  # transient sacct error; caller polls again
        for line in out.stdout.splitlines():
            if line.strip():  # first non-empty row = the primary job, not .batch/.extern
                return map_sacct_state(line)
        return Status.PENDING

    def logs(self, handle: Handle, *, tail: int | None = None) -> str:
        p = Path(handle.log_path or "")
        if not p.is_file():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        if tail is not None:
            return "\n".join(text.splitlines()[-tail:])
        return text

    def cancel(self, handle: Handle) -> None:
        if handle.job_id:
            subprocess.run(["scancel", str(handle.job_id)], capture_output=True)
