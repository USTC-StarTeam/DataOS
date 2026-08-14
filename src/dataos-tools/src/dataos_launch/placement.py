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


def partition_facts(env: dict, partition: str | None = None) -> dict[str, Any]:
    """The ``slurm:`` facts in effect for one partition.

    **``slurm.partitions.<name>.<key>`` overrides ``slurm.<key>``.** One rule, every key, so a
    fact that turns out to be per-partition is declared where it is true instead of patched
    where it is read.

    Almost everything a scheduler enforces is per-partition, and ``env.yaml`` used to state all
    of it flat, as if it described the machine. Each of these was found separately, and the first
    three were each patched separately before the pattern was visible:

    * ``nodes`` — the pool ``gpu[01-03]`` belongs to ``gpu``; ``debug`` is ``gpu05`` alone, so
      carrying the pool across gives ``--partition=debug --nodelist=gpu01,gpu02,gpu03``, two
      constraints with no intersection.
    * ``max_time`` — ``gpu`` allows 3 days, ``debug`` one. Clamping a debug job to the 3-day
      figure lets it through, and it then PENDs forever with ``PartitionTimeLimit``.
    * ``qos_available`` — ``debug`` accepts only ``debug_normal``. The account-wide list would
      have let a run ask for ``gpu_priv`` there and be refused by sbatch, with a message about
      the QOS rather than about the pairing.
    * ``gres_required`` — ``gpu05`` is heterogeneous (6x H100-80GB + 4x 40GB), so the debug
      partition is the one place a card model has to be named or a job OOMs at random.

    An environment with no ``partitions:`` block behaves exactly as before.
    """
    slurm = dict(env.get("slurm") or {})
    per_partition = slurm.pop("partitions", None) or {}
    name = partition or slurm.get("partition")
    return {**slurm, **(per_partition.get(name) or {}), "partition": name}


def partitions_available(env: dict) -> list[str]:
    """Which partitions this deployment declares — the keys of ``slurm.partitions``.

    Derived rather than listed: a separate ``partitions_available`` field would be a second
    place to add a partition, and therefore a place to forget one.
    """
    return sorted((env.get("slurm") or {}).get("partitions") or {})


def placement(env: dict, resource: dict | None = None) -> dict[str, Any]:
    """The launcher's kwargs: the run's choices over the environment's facts."""
    chosen = resource or {}
    # A partition is a choice like the QOS is, and for the same reason: it is per-RUN, not
    # per-machine. `debug` on PTB takes only the `debug_normal` QOS and caps a user at one job
    # and one GPU, so "run this smoke on debug" is a sentence about one run. Resolve it FIRST —
    # every fact below is read from that partition's block.
    facts = partition_facts(env, chosen.get("partition"))
    available = facts.get("qos_available") or []
    return {
        "partition": facts.get("partition"),
        # An environment DECLARES which QOS the partition allows; it does not pick one. The
        # fallbacks behind the run's choice are the env's own default and then, only so a
        # config that never mentioned QOS still submits, the first declared one.
        "qos": chosen.get("qos") or facts.get("qos") or (available[0] if available else None),
        # `nodes` is a POOL, not a pin. `run.resource.node` is the pin. Passing the pool as
        # --nodelist is deliberate: it constrains Slurm to nodes this deployment vouches for and
        # leaves the choice to the scheduler.
        "nodelist": chosen.get("node") or facts.get("nodes"),
        "account": chosen.get("account") or facts.get("account"),
        # Caps, so the launcher can clamp rather than let a job pend forever against a limit no
        # error message ever names.
        "max_time": facts.get("max_time"),
        "gres_required": facts.get("gres_required"),
        "mem_per_gpu_mb": facts.get("mem_per_gpu_mb"),
        "cpus_per_gpu": facts.get("cpus_per_gpu"),
        "guard_qos": facts.get("guard_qos"),
    }


def launcher_from_config(env: dict, resource: dict | None = None) -> Launcher:
    """``env.yaml`` + ``run.resource`` -> a launcher ready to take Jobs."""
    return get_launcher(launcher_name(env), **placement(env, resource))
