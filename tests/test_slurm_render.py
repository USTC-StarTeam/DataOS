"""Pure tests for SlurmLauncher sbatch rendering + sacct->Status mapping (no scheduler)."""

from __future__ import annotations

from dataos_train.launchers.slurm import SlurmLauncher, map_sacct_state
from dataos_train.types import JobSpec, LaunchPlan, Resources, SlurmExtras, Status


def _job(**kw) -> JobSpec:
    base = dict(
        backend="llamafactory",
        job_name="exp1",
        command=["llamafactory-cli", "train", "/run/train.yaml"],
        run_dir="/run",
        resources=Resources(nodes=1, gpus=2, cpus=8, time_h=1.5),
        conda_sh="/opt/conda/etc/profile.d/conda.sh",
        conda_env="Llama",
        pythonpath=["/deploy/src", "/deploy"],
        slurm=SlurmExtras(partition="gpu", qos="gpu_priv", nodelist="gpu04"),
    )
    base.update(kw)
    return JobSpec(**base)


def test_sbatch_contains_directives() -> None:
    text = SlurmLauncher().render_sbatch(_job())
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


def test_sbatch_gres_defaults_from_gpus() -> None:
    text = SlurmLauncher().render_sbatch(_job(slurm=SlurmExtras()))
    assert "#SBATCH --gres=gpu:2" in text  # derived from resources.gpus


def test_sbatch_torchrun() -> None:
    text = SlurmLauncher().render_sbatch(_job(launch=LaunchPlan(kind="torchrun")))
    assert "torchrun --nnodes 1 --nproc-per-node 2" in text


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
