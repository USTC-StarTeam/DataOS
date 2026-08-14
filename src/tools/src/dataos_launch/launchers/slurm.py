"""Slurm launcher: sbatch render + sacct poll.

Runs **locally on the cluster** (login node) — ssh/rsync stays in the DataOS deploy/sub-agent
layer, never inside the launcher.

This is where a ``Resources`` — a claim about what the program needs — becomes a partition's
dialect. The two are not the same language, and that translation is the job of this file:

* ``gres`` is not ``gpus``. A partition can force a card onto work that has none (PTB's does), so
  ``gpus=0`` still yields ``--gres=gpu:1`` — plus the GPU-Guard placeholder, because an idle card
  on the guarded QOS is exactly what gets a job scancelled.
* ``mem_gb`` is not ``--mem``. Where memory is granted per card, wanting more than one card's
  worth means wanting more cards; asking the other way round, sbatch rejects outright.
* ``time_h`` is not ``--time``. A request above the partition's MaxTime does not fail — it PENDs
  forever with ``PartitionTimeLimit``, which reads like a busy cluster.

Each of those was previously handled by each of the three CLIs that submit jobs, differently or
not at all.
"""

from __future__ import annotations

import re
import subprocess
import sys
from math import ceil
from pathlib import Path

from ..types import Handle, Job, Resources, Status
from .base import Launcher, build_preamble, compose_launch_line

_FAIL = {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "PREEMPTED", "REVOKED"}
_TIMEOUT = {"TIMEOUT", "DEADLINE"}
_RUNNING = {"RUNNING", "COMPLETING", "RESIZING", "SUSPENDED"}

#: What the GPU Guard demands of a job that does not itself use the GPU: a resident, *touched*
#: allocation above its memory floor, plus enough utilisation to clear its third rule. Both are
#: needed — the guard kills on any of peak memory, mean memory, or mean utilisation, and a
#: placeholder tensor alone satisfies the first two and fails the third. 30 GiB rather than the
#: nominal 24576 MiB threshold because under ``expandable_segments`` a nominal 26 GiB is accounted
#: as 24320 MiB; the 2026-08-11 batch lost 9 jobs at 24428/24320/24300/24296 MiB — all just under.
GUARD_GIB = 30
_GUARD_SCRIPT = """\
import threading, time, torch
hold = torch.empty({gib} * 2**30, dtype=torch.uint8, device="cuda").fill_(1)  # fill_ = touch it


def spin():
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)
    while True:
        torch.matmul(a, a, out=out)


threading.Thread(target=spin, daemon=True).start()
time.sleep(10**9)
"""


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


def slurm_time_s(v: str) -> int:
    """Slurm time -> seconds: ``[[days-]hours:]minutes[:seconds]``; a bare number is minutes."""
    days, _, rest = v.partition("-")
    if not rest:
        days, rest = "0", v
    if ":" not in rest:
        return int(days) * 86400 + int(rest) * 60
    secs = 0
    for part in rest.split(":"):
        secs = secs * 60 + int(part)
    return int(days) * 86400 + secs


def _hms(seconds: int) -> str:
    total = max(60, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _gres_count(gres: str) -> int:
    """``"gpu:1"`` / ``"gpu:a100:2"`` -> the trailing count; 0 when there is no number."""
    tail = str(gres).rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _gres_kind(gres: str | None) -> str:
    """The part before the count: ``"gpu"``, or ``"gpu:nvidia_h100_80gb_hbm3"``.

    A node can be heterogeneous, and one of PTB's is: gpu05 carries
    ``gpu:nvidia_h100_80gb_hbm3:6`` alongside ``gpu:40gb:4``. An untyped ``--gres=gpu:1`` may
    land on either, so a full-SFT sized for 80 GiB OOMs on roughly half its submissions and
    succeeds on the rest — a failure that looks like a flaky cluster rather than a request that
    never said what it needed.
    """
    if not gres:
        return "gpu"
    parts = str(gres).split(":")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return ":".join(parts) or "gpu"


class SlurmLauncher(Launcher):
    name = "slurm"
    script_suffix = ".sbatch"

    def __init__(
        self,
        *,
        partition: str | None = None,
        qos: str | None = None,
        nodelist: str | None = None,
        account: str | None = None,
        max_time: str | None = None,
        gres_required: str | None = None,
        mem_per_gpu_mb: int | None = None,
        cpus_per_gpu: int | None = None,
        guard_qos: str | None = None,
        extra_sbatch: list[str] | None = None,
        **_deployment: object,  # a deployment declares more than any one launcher reads
    ) -> None:
        self.partition = partition
        self.qos = qos
        self.nodelist = nodelist
        self.account = account
        self.max_time = max_time
        self.gres_required = gres_required
        self.mem_per_gpu_mb = mem_per_gpu_mb
        self.cpus_per_gpu = cpus_per_gpu
        self.guard_qos = guard_qos
        self.extra_sbatch = list(extra_sbatch or [])

    # --- Resources -> partition dialect --------------------------------------------------

    def cards(self, r: Resources) -> int:
        """How many GPUs to ASK for, which is not how many the job uses."""
        want = r.gpus
        if r.mem_gb and self.mem_per_gpu_mb:
            want = max(want, ceil(r.mem_gb * 1024 / self.mem_per_gpu_mb))
        if self.gres_required:
            want = max(want, _gres_count(self.gres_required))
        return want

    def cpus(self, r: Resources) -> int:
        """``--cpus-per-task``, clamped to what the cards bought.

        Over the cap this partition does not reject the job — it pends, and a pending job is
        indistinguishable from a busy cluster until someone reads ``squeue -o %R``.
        """
        cap = (self.cpus_per_gpu or 0) * self.cards(r)
        if cap and r.cpus > cap:
            print(
                f"dataos-launch: --cpus-per-task {r.cpus} exceeds {self.cpus_per_gpu}/GPU x "
                f"{self.cards(r)} GPU(s); clamped to {cap}",
                file=sys.stderr,
            )
            return cap
        return r.cpus

    def time_limit(self, r: Resources) -> str:
        """``--time``, clamped to the partition maximum.

        Always emitted: with no ``--time`` sbatch falls back to the QOS limit (7d here), which
        EXCEEDS the partition MaxTime (3d), and the job then pends forever rather than failing.
        """
        seconds = int(round(r.time_h * 3600))
        if self.max_time:
            limit = slurm_time_s(self.max_time)
            if seconds > limit:
                print(
                    f"dataos-launch: --time {_hms(seconds)} exceeds the partition max "
                    f"{self.max_time}; clamped",
                    file=sys.stderr,
                )
                seconds = limit
        return _hms(seconds)

    def needs_guard(self, r: Resources) -> bool:
        """A job holding a card it does not use, on the QOS that watches for exactly that."""
        return r.gpus == 0 and bool(self.guard_qos) and self.qos == self.guard_qos

    # --- render / dispatch ---------------------------------------------------------------

    def render(self, job: Job) -> str:
        r = job.resources
        cards = self.cards(r)
        lines = [
            "#!/bin/bash",
            f"#SBATCH -J {job.job_name}",
            "#SBATCH -o slurm-%j.out",
            "#SBATCH -e slurm-%j.out",
            f"#SBATCH --nodes={r.nodes}",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={self.cpus(r)}",
            f"#SBATCH --time={self.time_limit(r)}",
        ]
        if cards:
            # The KIND comes from the environment's own declaration, so a deployment that must
            # name a card model gets to; the COUNT is this job's.
            lines.append(f"#SBATCH --gres={_gres_kind(self.gres_required)}:{cards}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")
        if self.nodelist:
            lines.append(f"#SBATCH --nodelist={self.nodelist}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if r.mem_gb:
            lines.append(f"#SBATCH --mem={r.mem_gb}G")
        for extra in self.extra_sbatch:
            lines.append(extra if extra.lstrip().startswith("#SBATCH") else f"#SBATCH {extra}")
        lines.append("")
        lines.extend(build_preamble(job))
        if self.needs_guard(r):
            # After the preamble, so `python` is the job's own env — the one that has torch — and
            # backgrounded, because it never returns. No trap reaps it: Slurm tears down the job's
            # cgroup at the end, and a second EXIT trap here would silently replace whichever one
            # `job.pre_cmds` installed (the eval bridge installs one, to remove TMPDIR).
            lines.append("cat > .gpu_guard.py <<'DATAOS_GUARD_EOF'")
            lines.append(_GUARD_SCRIPT.format(gib=GUARD_GIB).rstrip())
            lines.append("DATAOS_GUARD_EOF")
            lines.append("python .gpu_guard.py &   # GPU Guard placeholder; see needs_guard()")
        lines.append(compose_launch_line(job))
        return "\n".join(lines) + "\n"

    def dispatch(self, job: Job, script: Path) -> Handle:
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
        return self.new_handle(
            job, job_id=job_id, log_path=str(Path(job.run_dir) / f"slurm-{job_id}.out")
        )

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

    def cancel(self, handle: Handle) -> None:
        subprocess.run(["scancel", str(handle.job_id)], capture_output=True)
