"""Build the launcher the deployment names, placed where the run chose.

The one place the three config layers meet. ``configs/environments/<env>/env.yaml`` says what the
machine HAS — which partition, which QOS exist, which nodes, the per-card caps. ``configs/runs/
*.json`` says what this run CHOSE — ``resource.qos``, ``resource.node``, ``resource.gpus``. The
rule between them is a single sentence, *run first, environment second*, and it was written three
times: once in dataos-train, once in dataos-eval, once in dataos-data.

Three copies of a one-line rule is not a style problem. When ``qos`` moved from ``env.yaml`` into
the run config, dataos-train's copy went on reading the field that had just vanished, so its jobs
went out with no ``--qos`` at all and Slurm applied the account default — the *guarded* queue —
while their own prompt told them they were on an unguarded one. Nothing errored. On the run
before that, a run pinned to ``gpu02`` put 35 of 35 training jobs there and 68 of 94 eval jobs
somewhere else, because dataos-eval's copy read the environment's node POOL and never saw the
pin. Both bugs are unrepresentable if there is one copy, which is this one.

What a stage passes is its ``Resources`` — what the program needs. What the launcher decides is
placement — where that runs. Neither gets a vote on the other.
"""

from __future__ import annotations

from typing import Any

from .launchers.base import Launcher
from .registry import get_launcher


def launcher_name(env: dict) -> str:
    """Which launcher runs here — declared by the environment, never guessed.

    Defaults to ``slurm`` because every environment predating the field is a Slurm cluster; a
    workstation says ``launcher: local`` and the same Job then runs as a subprocess. That is the
    whole point of the split: what a stage does is not a function of the machine, only what
    starts it is.
    """
    slurm = env.get("slurm") or {}
    default = "local" if env.get("kind") == "local" else "slurm"
    return str(slurm.get("launcher") or env.get("launcher") or default)


def placement(env: dict, resource: dict | None = None) -> dict[str, Any]:
    """The launcher's kwargs: the run's choices over the environment's facts."""
    slurm = env.get("slurm") or {}
    chosen = resource or {}
    available = slurm.get("qos_available") or []
    return {
        "partition": slurm.get("partition"),
        # An environment DECLARES which QOS the account may use; it does not pick one. The
        # fallbacks behind the run's choice are the env's own default and then, only so a
        # config that never mentioned QOS still submits, the first declared one.
        "qos": chosen.get("qos") or slurm.get("qos") or (available[0] if available else None),
        # `env.slurm.nodes` is a POOL ("gpu01,gpu02,gpu03"), not a pin. `run.resource.node` is
        # the pin. Passing the pool as --nodelist is correct and deliberate: it constrains
        # Slurm to nodes this deployment vouches for, and leaves the choice to the scheduler.
        "nodelist": chosen.get("node") or slurm.get("nodes"),
        "account": chosen.get("account") or slurm.get("account"),
        # Caps: facts about the partition, so the launcher can clamp rather than let a job pend
        # forever on a limit no error message ever names.
        "max_time": slurm.get("max_time"),
        "gres_required": slurm.get("gres_required"),
        "mem_per_gpu_mb": slurm.get("mem_per_gpu_mb"),
        "cpus_per_gpu": slurm.get("cpus_per_gpu"),
        "guard_qos": slurm.get("guard_qos"),
    }


def launcher_from_config(env: dict, resource: dict | None = None) -> Launcher:
    """``env.yaml`` + ``run.resource`` -> a launcher ready to take Jobs."""
    return get_launcher(launcher_name(env), **placement(env, resource))
