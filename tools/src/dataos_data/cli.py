"""dataos-data CLI: submit / status for the data stage.

The third instrument, and the thinnest. ``dataos-train`` owns a spec with a closed schema and can
validate its INPUT; ``dataos-eval`` drives a frozen protocol and takes almost no input at all.
This one takes arbitrary Python — the recipe IS the method DataOS searches, so schematising it
would be the one mistake worth avoiding. What it owns instead is everything around the recipe:

* the job shell. Until now every node hand-wrote it, re-deriving the same environment lessons the
  other two tools already encode — short TMPDIR (the run-root path exceeds the 108-char AF_UNIX
  socket limit), cache redirects out of the shared HOME, offline HF, venv activation, the plugin
  path for run-local operators. Copied by hand, per node, from memory.
* the output contract, appended to the job as its last step so a build that produced nothing fails
  the build (see :mod:`dataos_data.verify`).
* the manifest's name, which the recipe kept inventing.

It does NOT depend on dataos-operator, for the same reason dataos-train does not depend on
LLaMA-Factory: the operators are the substrate the recipe is written in, they live in the exec
host's venv, and an instrument that drives a substrate should not import it.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml
from dataos_launch import Handle, Job, Resources, Status, launcher_from_config, placement
from dataos_launch.launchers.slurm import slurm_time_s

_HANDLE = "data_handle.json"


def _load_env(repo: Path, environment: str) -> dict:
    return yaml.safe_load(
        (repo / "configs" / "environments" / environment / "env.yaml").read_text(encoding="utf-8")
    )


def _build_job(args, env: dict) -> Job:
    """The recipe, wrapped in everything the recipe should not have to know."""
    run_root = shlex.quote(args.run_root)
    venv = env["tools"]["sft_env"]["path"]
    recipe = Path(args.recipe).name  # the recipe is submitted from inside run_dir

    plugin_dirs = [f"{args.run_root}/custom_operators"]
    if args.shared_operators:
        plugin_dirs.insert(0, args.shared_operators)

    job_env = {
        # Redirect every cache out of the shared HOME and into the run root, so a node's work is
        # self-contained and a shared filesystem is not the place two runs collide.
        "HF_HOME": f"{args.run_root}/cache/hf",
        "HF_DATASETS_CACHE": f"{args.run_root}/cache/hf/datasets",
        "TORCH_HOME": f"{args.run_root}/cache/torch",
        # The pool and the benchmarks are on disk; a compute node has no internet, and an
        # accidental hub call there fails slowly and confusingly rather than fast.
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        # Operators registered per run, plus whatever the deployment ships. Writing a run-local
        # operator is routine, and this is the only thing that makes one resolvable by name.
        "DATAOS_OPERATOR_PLUGINS": ":".join(plugin_dirs),
    }

    pre_cmds = [
        # A run-root-relative TMPDIR exceeds the 108-character AF_UNIX socket limit; short path.
        'export TMPDIR=/tmp/job_${SLURM_JOB_ID:-$$}',
        f'mkdir -p "$TMPDIR" {run_root}/cache/hf {run_root}/cache/torch',
        "trap 'rm -rf \"$TMPDIR\"' EXIT",
    ]

    py = f"{venv}/bin/python"
    verify = [
        py, "-m", "dataos_data.verify",
        "--dir", f"{args.run_root}/data/sft",
    ]
    if args.stats:
        verify += ["--stats", args.stats]

    return Job(
        backend="dataos-data",
        job_name=f"data_{Path(args.run_root).name}"[:64],
        # The recipe runs, then the contract is checked in the same job — so "built nothing"
        # is a failed build rather than a training job that dies on an empty dataset.
        command=["bash", "-lc", f"{shlex.quote(py)} {shlex.quote(recipe)} && {shlex.join(verify)}"],
        env=job_env,
        run_dir=args.run_root,
        conda_env=venv,
        resources=Resources(gpus=args.gpus, cpus=args.cpus,
                            time_h=slurm_time_s(args.time) / 3600),
        pre_cmds=pre_cmds,
    )


def _resource(args) -> dict:
    """The run's placement choices, as ``configs/runs/*.json`` states them.

    These arrive as CLI flags because the agent invokes the tool over ssh, but they are the run
    config's ``resource`` block and nothing else; the precedence against ``env.yaml`` lives in
    :func:`dataos_launch.placement.placement`, once, for all three instruments.
    """
    return {"qos": args.qos, "node": args.nodelist}


def cmd_submit(args) -> int:
    repo = Path(args.repo).resolve()
    env = _load_env(repo, args.environment)
    job = _build_job(args, env)
    launcher = launcher_from_config(env, _resource(args))
    if args.dry_run:
        # Placement is the launcher's now, but a dry run still has to answer "what gets
        # submitted" — and half that answer would otherwise be invisible.
        print(json.dumps({"launcher": launcher.name,
                          "placement": placement(env, _resource(args)),
                          "job": job.model_dump()},
                         ensure_ascii=False, indent=1, default=str))
        return 0
    handle = launcher.submit(job)
    path = Path(args.run_root) / _HANDLE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(handle.model_dump(), ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(handle.model_dump(), ensure_ascii=False, default=str))
    return 0


def cmd_status(args) -> int:
    repo = Path(args.repo).resolve()
    env = _load_env(repo, args.environment)
    handle = Handle.model_validate(json.loads(Path(args.handle).read_text(encoding="utf-8")))
    state = launcher_from_config(env, _resource(args)).status(handle)
    print(json.dumps({"status": state.value if isinstance(state, Status) else str(state)},
                     ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataos-data")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit")
    s.add_argument("--repo", required=True, help="deploy checkout root (configs/ lives there)")
    s.add_argument("--environment", default="PTB_H100_slurm")
    s.add_argument("--run-root", required=True, help="the node's run root; the recipe lives here")
    s.add_argument("--recipe", required=True, help="the recipe script, inside run-root")
    s.add_argument("--stats", default=None,
                   help="optional JSON the recipe writes with its own per-source counts")
    s.add_argument("--shared-operators", default=None,
                   help="deployment-wide custom operator dir, prepended to the plugin path")
    # The run's resource facts, same four the other two instruments take.
    s.add_argument("--gpus", type=int, default=0,
                   help="GPUs the RECIPE uses; 0 (the default — building data is CPU\n                        work) still gets the card this partition forces, plus the\n                        GPU-Guard placeholder when the QOS is watched")
    s.add_argument("--cpus", type=int, default=8)
    s.add_argument("--qos", default=None, help="run-level QOS; env.yaml declares which exist")
    s.add_argument("--nodelist", default=None, help="run-level node pin; overrides the env pool")
    s.add_argument("--time", default="04:00:00")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_submit)

    t = sub.add_parser("status")
    t.add_argument("--repo", required=True)
    t.add_argument("--environment", default="PTB_H100_slurm")
    t.add_argument("handle")
    t.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
