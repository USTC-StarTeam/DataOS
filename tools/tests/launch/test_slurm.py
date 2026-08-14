"""SlurmLauncher: sbatch rendering, the Resources->partition translation, sacct->Status.

No scheduler is involved. The translation cases are the point of the file — each one is a way a
job used to die quietly: pending forever against a limit no message names, or scancelled by a
guard whose kill says nothing about itself.
"""

from __future__ import annotations

import stat

from dataos_launch.launchers.slurm import SlurmLauncher, map_sacct_state
from dataos_launch.types import Handle, Job, LaunchPlan, Resources, Status

PTB = dict(  # the PTB_H100_slurm facts, as env.yaml states them
    partition="gpu",
    qos="gpu_priv",
    nodelist="gpu04",
    max_time="3-00:00:00",
    gres_required="gpu:1",
    mem_per_gpu_mb=125000,
    cpus_per_gpu=11,
    guard_qos="gpu_normal",
)


def _job(**kw) -> Job:
    base = dict(
        backend="llamafactory",
        job_name="exp1",
        command=["llamafactory-cli", "train", "/run/train.yaml"],
        run_dir="/run",
        resources=Resources(nodes=1, gpus=2, cpus=8, time_h=1.5),
        conda_sh="/opt/conda/etc/profile.d/conda.sh",
        conda_env="Llama",
        pythonpath=["/deploy/src", "/deploy"],
    )
    base.update(kw)
    return Job(**base)


def test_sbatch_contains_directives() -> None:
    text = SlurmLauncher(**PTB).render(_job())
    for snippet in (
        "#SBATCH -J exp1",
        "#SBATCH --partition=gpu",
        "#SBATCH --qos=gpu_priv",
        "#SBATCH --gres=gpu:2",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --nodelist=gpu04",
        "#SBATCH --time=01:30:00",
        "source /opt/conda/etc/profile.d/conda.sh && conda activate Llama",
        "export PYTHONPATH=/deploy/src:/deploy",
        "llamafactory-cli train /run/train.yaml",
    ):
        assert snippet in text, f"missing: {snippet}\n---\n{text}"


def test_placement_comes_from_config_not_from_the_job() -> None:
    """A Job has no scheduler vocabulary to state placement WITH. That is the invariant."""
    assert not hasattr(_job(), "slurm")
    text = SlurmLauncher(partition="debug", qos="gpu_normal").render(_job())
    assert "#SBATCH --partition=debug" in text and "#SBATCH --qos=gpu_normal" in text


def test_gres_is_not_gpus() -> None:
    """A CPU job still gets the card this partition forces onto everything."""
    text = SlurmLauncher(**PTB).render(_job(resources=Resources(gpus=0, cpus=8)))
    assert "#SBATCH --gres=gpu:1" in text


def test_a_typed_gres_keeps_its_type() -> None:
    """A node can be heterogeneous, and PTB's gpu05 is: 6x nvidia_h100_80gb_hbm3 + 4x 40gb.
    An untyped `--gres=gpu:1` lands on either, so a run sized for 80 GiB OOMs on some
    submissions and not others — which reads as a flaky cluster, not as an underspecified
    request. The KIND is the environment's declaration; only the COUNT is the job's."""
    launcher = SlurmLauncher(**{**PTB, "gres_required": "gpu:nvidia_h100_80gb_hbm3:1"})
    text = launcher.render(_job(resources=Resources(gpus=2)))
    assert "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2" in text
    # and a CPU job still gets the one card the partition forces, of that same type
    cpu = launcher.render(_job(resources=Resources(gpus=0, cpus=8)))
    assert "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1" in cpu


def test_no_gres_where_the_partition_does_not_require_one() -> None:
    text = SlurmLauncher(partition="cpu").render(_job(resources=Resources(gpus=0, cpus=8)))
    assert "--gres" not in text


def test_memory_buys_cards() -> None:
    """--mem above one card's grant is refused by sbatch; ask for the cards it implies."""
    text = SlurmLauncher(**PTB).render(_job(resources=Resources(gpus=1, mem_gb=200)))
    assert "#SBATCH --gres=gpu:2" in text  # 200 GiB against 125000 MiB per card


def test_cpus_clamped_to_what_the_cards_bought() -> None:
    """Over the per-GPU cap this partition does not reject the job — it pends, indefinitely."""
    text = SlurmLauncher(**PTB).render(_job(resources=Resources(gpus=1, cpus=32)))
    assert "#SBATCH --cpus-per-task=11" in text


def test_time_clamped_to_partition_max() -> None:
    """Above MaxTime a job pends with PartitionTimeLimit rather than failing."""
    text = SlurmLauncher(**PTB).render(_job(resources=Resources(time_h=100)))
    assert "#SBATCH --time=72:00:00" in text


def test_guard_installed_only_for_an_idle_card_on_the_watched_qos() -> None:
    cpu_job = _job(resources=Resources(gpus=0, cpus=8))
    watched = SlurmLauncher(**{**PTB, "qos": "gpu_normal"})
    text = watched.render(cpu_job)
    assert "gpu_guard" in text and "torch.empty(30 * 2**30" in text and "matmul" in text

    # ...not on the unguarded QOS, where it only takes capacity from the work,
    assert "gpu_guard" not in SlurmLauncher(**PTB).render(cpu_job)
    # ...and not for a job that uses the card itself.
    assert "gpu_guard" not in watched.render(_job(resources=Resources(gpus=1)))


def test_sbatch_torchrun() -> None:
    text = SlurmLauncher(**PTB).render(_job(launch=LaunchPlan(kind="torchrun")))
    assert "torchrun --nnodes 1 --nproc-per-node 2" in text


def test_status_reads_the_primary_job_row(tmp_path, monkeypatch) -> None:
    """sacct lists the main job first, then .batch/.extern — a failed job's .extern is
    COMPLETED, so reading the LAST row misreports FAILED as COMPLETED. Came here from
    dataos-eval's suite when the polling moved into the launcher."""
    fake = tmp_path / "sacct"
    fake.write_text("#!/bin/bash\necho FAILED; echo COMPLETED; echo COMPLETED\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    handle = Handle(launcher="slurm", spec_id="s1", run_dir=str(tmp_path), job_name="exp1",
                    submitted_at="2026-08-13T00:00:00Z", job_id="1")
    assert SlurmLauncher().status(handle) == Status.FAILED


def test_sacct_mapping() -> None:
    assert map_sacct_state("COMPLETED") == Status.COMPLETED
    assert map_sacct_state("  COMPLETED ") == Status.COMPLETED
    assert map_sacct_state("CANCELLED by 1001") == Status.FAILED
    assert map_sacct_state("TIMEOUT") == Status.TIMEOUT
    assert map_sacct_state("FAILED") == Status.FAILED
    assert map_sacct_state("OUT_OF_MEMORY") == Status.FAILED
    assert map_sacct_state("RUNNING") == Status.RUNNING
    assert map_sacct_state("PENDING") == Status.PENDING
    assert map_sacct_state("") == Status.PENDING
