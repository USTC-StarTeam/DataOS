"""The data stage: a recipe, wrapped in everything the recipe should not have to know.

The thinnest of the three specs, deliberately. ``TrainSpec`` closes over a method schema because
the method is what DataOS searches; the data recipe is ALSO what DataOS searches, and it is
arbitrary Python — schematising it would be the one mistake worth avoiding here. So this spec
names the recipe and says nothing about what is in it.

What it does own is everything around the recipe, which every node used to hand-write from
memory: the short TMPDIR (a run-root-relative one exceeds the 108-character AF_UNIX socket
limit), the cache redirects out of the shared HOME, offline HF, venv activation, the plugin path
that makes a run-local operator resolvable by name, and the output contract appended as the
job's last step so a build that produced nothing fails the BUILD rather than the training job
downstream.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from dataos_launch import Job
from dataos_launch.spec import Spec
from dataos_launch.stage import Stage


class DataSpec(Spec):
    """One recipe, run in this deployment's data environment.

    ``resources.gpus`` defaults to 0 here and means it: building data is CPU work. The partition
    may still force a card, and on the guarded QOS the launcher installs the placeholder that
    keeps the guard from scancelling a job for holding one idle.
    """

    stage = "data"

    recipe: str  # file name, resolved inside run_dir — the job runs with cwd=run_dir
    stats: str | None = None  # optional JSON the recipe writes with per-source counts
    shared_operators: str | None = None  # deployment-wide operator dir, prepended to the path
    #: Whether this script is contracted to PRODUCE the dataset. False makes it a probe: same
    #: job shell, no output contract, nothing appended. An experimenter that wants to look at a
    #: pool's schema before writing its recipe has to run python on a compute node, and with no
    #: way to say "this one produces nothing" it hand-wrote an sbatch and re-derived the short
    #: TMPDIR, the offline-HF flags, the cache redirects and the venv activation this stage
    #: already owns — the exact boilerplate the stage exists to stop anyone rewriting.
    verify: bool = True


class DataStage(Stage):
    """Runs a recipe and checks what it produced, in one job."""

    name = "data"

    @property
    def venv(self) -> str:
        return self.env["tools"]["sft_env"]["path"]

    def validate(self, spec: DataSpec) -> list[str]:
        errors = []
        recipe = Path(spec.run_dir) / Path(spec.recipe).name
        if not recipe.is_file():
            errors.append(f"recipe not found in the run root: {recipe}")
        if not Path(self.venv).is_dir():
            errors.append(f"the environment's sft_env does not exist: {self.venv}")
        return errors

    def build(self, spec: DataSpec) -> Job:
        run_root = shlex.quote(spec.run_dir)
        recipe = Path(spec.recipe).name  # the recipe is submitted from inside run_dir

        plugin_dirs = [f"{spec.run_dir}/custom_operators"]
        if spec.shared_operators:
            plugin_dirs.insert(0, spec.shared_operators)

        job_env = {
            # Redirect every cache out of the shared HOME and into the run root, so a node's work
            # is self-contained and a shared filesystem is not the place two runs collide.
            "HF_HOME": f"{spec.run_dir}/cache/hf",
            "HF_DATASETS_CACHE": f"{spec.run_dir}/cache/hf/datasets",
            "TORCH_HOME": f"{spec.run_dir}/cache/torch",
            # The pool and the benchmarks are on disk; a compute node has no internet, and an
            # accidental hub call there fails slowly and confusingly rather than fast.
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            # Operators registered per run, plus whatever the deployment ships. Writing a
            # run-local operator is routine, and this is the only thing that resolves one by name.
            "DATAOS_OPERATOR_PLUGINS": ":".join(plugin_dirs),
        }
        pre_cmds = [
            "export TMPDIR=/tmp/job_${SLURM_JOB_ID:-$$}",
            f'mkdir -p "$TMPDIR" {run_root}/cache/hf {run_root}/cache/torch',
            "trap 'rm -rf \"$TMPDIR\"' EXIT",
        ]

        py = f"{self.venv}/bin/python"
        line = shlex.quote(py) + " " + shlex.quote(recipe)
        if spec.verify:
            # The recipe runs, then the contract is checked in the same job — so "built nothing"
            # is a failed build rather than a training job that dies on an empty dataset.
            verify = [py, "-m", "dataos_data.verify", "--dir", f"{spec.run_dir}/data/sft"]
            if spec.stats:
                verify += ["--stats", spec.stats]
            line += " && " + shlex.join(verify)

        return Job(
            backend="dataos-data",
            job_name=spec.name,
            command=["bash", "-lc", line],
            env=job_env,
            run_dir=spec.run_dir,
            conda_env=self.venv,
            resources=spec.resources,
            pre_cmds=pre_cmds,
        )
