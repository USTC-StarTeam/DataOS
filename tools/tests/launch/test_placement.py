"""Placement: the run's choices over the environment's facts, resolved in one place.

Both regressions below shipped, ran a full batch, and were found afterwards in the logs. Neither
raised anything: a job with no ``--qos`` submits fine, and a job that ignores a node pin lands on
a perfectly good node. They are here because the failure mode of this rule is silence.
"""

from __future__ import annotations

from dataos_launch.placement import (
    launcher_from_config,
    launcher_name,
    partition_facts,
    partitions_available,
    placement,
)

PTB_ENV = {
    "name": "PTB_H100_slurm",
    "kind": "slurm",
    "slurm": {
        "partition": "gpu",
        "qos": "gpu_priv",
        # machine-wide, inherited by every partition
        "gres_required": "gpu:1",
        "mem_per_gpu_mb": 125000,
        "cpus_per_gpu": 11,
        "guard_qos": "gpu_normal",
        "partitions": {
            "gpu": {
                "nodes": "gpu01,gpu02,gpu03",
                "max_time": "3-00:00:00",
                "qos_available": ["gpu_priv", "gpu_normal"],
            },
            "debug": {
                "nodes": "gpu05",
                "max_time": "1-00:00:00",
                "qos_available": ["debug_normal"],
                "gres_required": "gpu:nvidia_h100_80gb_hbm3:1",
            },
        },
    },
}


def test_run_choice_wins_over_the_environment() -> None:
    p = placement(PTB_ENV, {"qos": "gpu_normal", "node": "gpu02"})
    assert p["qos"] == "gpu_normal"
    assert p["nodelist"] == "gpu02"


def test_environment_default_when_the_run_chose_nothing() -> None:
    p = placement(PTB_ENV, {})
    assert p["qos"] == "gpu_priv"
    # env.slurm.nodes is a POOL, not a pin — passing it constrains Slurm to nodes this
    # deployment vouches for and leaves the pick to the scheduler.
    assert p["nodelist"] == "gpu01,gpu02,gpu03"


def test_qos_never_falls_through_to_the_account_default() -> None:
    """When `qos` moved into the run config, one submitter went on reading the env field that
    had just vanished: its jobs went out with no --qos at all and Slurm applied the account
    default — the GUARDED queue — while their own prompt said unguarded. Nothing errored."""
    env = {"slurm": {"partition": "gpu", "qos_available": ["gpu_priv", "gpu_normal"]}}
    assert placement(env, {})["qos"] == "gpu_priv"
    assert placement(env, None)["qos"] == "gpu_priv"


def test_partition_caps_reach_the_launcher() -> None:
    """They are facts about the machine, so the launcher can clamp instead of letting a job
    pend forever against a limit no error message names."""
    p = placement(PTB_ENV, {})
    assert (p["max_time"], p["cpus_per_gpu"], p["mem_per_gpu_mb"]) == ("3-00:00:00", 11, 125000)
    assert (p["gres_required"], p["guard_qos"]) == ("gpu:1", "gpu_normal")


def test_launcher_name_is_declared_not_guessed() -> None:
    assert launcher_name(PTB_ENV) == "slurm"
    assert launcher_name({"kind": "local"}) == "local"
    assert launcher_name({"launcher": "local", "kind": "slurm"}) == "local"
    assert launcher_name({}) == "slurm"  # every env predating the field is a Slurm cluster


def test_local_launcher_accepts_and_ignores_placement() -> None:
    """One call site serves both deployments; a workstation has no partition to speak of."""
    launcher = launcher_from_config({**PTB_ENV, "launcher": "local"}, {"qos": "gpu_normal"})
    assert launcher.name == "local"


def test_from_config_builds_a_placed_launcher() -> None:
    launcher = launcher_from_config(PTB_ENV, {"qos": "gpu_normal", "node": "gpu02"})
    assert (launcher.name, launcher.qos, launcher.nodelist) == ("slurm", "gpu_normal", "gpu02")


# --- per-partition facts ---------------------------------------------------------------------


def test_a_partition_overrides_what_it_must_and_inherits_the_rest() -> None:
    """One rule for every key: `slurm.partitions.<name>.<key>` beats `slurm.<key>`.

    Everything below was first patched separately at the point it was READ, which is how the
    same mistake could recur four times. Declared where it is true, it cannot.
    """
    gpu, debug = partition_facts(PTB_ENV, "gpu"), partition_facts(PTB_ENV, "debug")
    # overridden
    assert (gpu["max_time"], debug["max_time"]) == ("3-00:00:00", "1-00:00:00")
    assert (gpu["nodes"], debug["nodes"]) == ("gpu01,gpu02,gpu03", "gpu05")
    assert gpu["qos_available"] == ["gpu_priv", "gpu_normal"]
    assert debug["qos_available"] == ["debug_normal"]
    assert debug["gres_required"] == "gpu:nvidia_h100_80gb_hbm3:1"   # gpu05 is heterogeneous
    # inherited
    assert gpu["gres_required"] == "gpu:1"
    for f in (gpu, debug):
        assert (f["mem_per_gpu_mb"], f["cpus_per_gpu"], f["guard_qos"]) == (125000, 11, "gpu_normal")


def test_no_partition_named_means_the_default_one() -> None:
    assert partition_facts(PTB_ENV)["partition"] == "gpu"
    assert partition_facts(PTB_ENV)["max_time"] == "3-00:00:00"


def test_an_environment_without_partitions_is_unchanged() -> None:
    """Every env that predates the block reads exactly as it did."""
    flat = {"slurm": {"partition": "gpu", "max_time": "3-00:00:00", "nodes": "gpu01"}}
    assert partition_facts(flat) == {"partition": "gpu", "max_time": "3-00:00:00", "nodes": "gpu01"}
    assert partitions_available(flat) == []


def test_the_partition_list_is_derived_not_declared() -> None:
    """A second place to add a partition is a place to forget one."""
    assert partitions_available(PTB_ENV) == ["debug", "gpu"]


def test_the_limits_a_run_is_under_follow_its_partition() -> None:
    """The bug this whole structure exists to make unrepresentable: a debug run clamped to the
    gpu partition's 3-day limit is let through and then PENDs forever with PartitionTimeLimit."""
    on_debug = placement(PTB_ENV, {"partition": "debug", "qos": "debug_normal"})
    assert on_debug["max_time"] == "1-00:00:00"
    assert on_debug["nodelist"] == "gpu05"          # not the gpu pool, which debug does not hold
    assert on_debug["gres_required"] == "gpu:nvidia_h100_80gb_hbm3:1"
