"""Shared launcher logic: asset materialization, bash preamble, LaunchPlan composition.

The launcher composes the actual exec line from ``job.launch.kind`` — the single place
launch topology is interpreted — so launchers never need to know which backend produced
the job.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from ..io import atomic_write_text
from ..types import JobSpec


class LaunchError(RuntimeError):
    pass


def materialize(job: JobSpec) -> Path:
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


def build_preamble(job: JobSpec) -> list[str]:
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


def compose_launch_line(job: JobSpec) -> str:
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
    if plan.kind == "ray":
        return _compose_ray(job)
    raise LaunchError(f"unknown launch kind: {plan.kind!r}")


def _compose_ray(job: JobSpec) -> str:
    quoted = " ".join(shlex.quote(c) for c in job.command)
    ray = job.launch.ray
    if ray is None or (job.resources.nodes or 1) <= 1:
        # Single node: verl starts a local Ray cluster in-process; just run the trainer.
        return quoted
    # Multi-node (best-effort; gated behind needs_slurm tests): start ray on every node,
    # then run the trainer once on the head.
    head = " ".join(ray.start_head_cmd or ["ray", "start", "--head", "--port", "6379"])
    join = " ".join(ray.worker_join_cmd or ["ray", "start", "--address", "$RAY_HEAD:6379"])
    return (
        'RAY_HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)\n'
        f'if [ "$SLURMD_NODENAME" = "$RAY_HEAD" ]; then {head}; else {join}; fi\n'
        f'if [ "$SLURMD_NODENAME" = "$RAY_HEAD" ]; then {quoted}; fi'
    )
