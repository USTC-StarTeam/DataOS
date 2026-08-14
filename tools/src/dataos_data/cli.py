"""dataos-data CLI: submit / status for the data stage.

The third instrument, and the thinnest. ``dataos-train`` owns a spec with a closed method schema
and can validate its INPUT; ``dataos-eval`` drives a frozen protocol and takes almost no input at
all. This one takes arbitrary Python — the recipe IS the method DataOS searches — and what it
owns instead is everything around it. That lives in :mod:`dataos_data.stage`; this file parses
flags and prints.

It does NOT depend on dataos-operator, for the same reason dataos-train does not depend on
LLaMA-Factory: the operators are the substrate the recipe is written in, they live in the exec
host's venv, and an instrument that drives a substrate should not import it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataos_launch import Handle, Resources
from dataos_launch.launchers.slurm import slurm_time_s
from dataos_launch.spec import SpecError

from .stage import DataSpec, DataStage

_HANDLE = "data_handle.json"


def cmd_submit(args) -> int:
    probe = getattr(args, "probe", False)
    stage = DataStage.from_repo(Path(args.repo).resolve(), args.environment)
    spec = DataSpec(
        name=f"{'probe' if probe else 'data'}_{Path(args.run_root).name}"[:64],
        run_dir=args.run_root,
        resources=Resources(
            gpus=args.gpus, cpus=args.cpus, time_h=slurm_time_s(args.time) / 3600
        ),
        recipe=args.recipe,
        stats=args.stats,
        shared_operators=args.shared_operators,
        verify=not probe,
    )
    # The run config's `resource` block, arriving as flags because the agent calls this over ssh.
    resource = {"qos": args.qos, "node": args.nodelist, "partition": args.partition}

    if args.dry_run:
        print(json.dumps(stage.preview(spec, resource), ensure_ascii=False, indent=1,
                         default=str))
        return 0
    try:
        handle = stage.submit(spec, resource)
    except SpecError as exc:
        print(json.dumps({"ok": False, "errors": exc.errors}), file=sys.stderr)
        return 1
    spec.write()
    path = Path(spec.run_dir) / _HANDLE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(handle.model_dump(), ensure_ascii=False, default=str)
    path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def _handle(path: str) -> Handle:
    return Handle.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def cmd_status(args) -> int:
    """Polling needs no placement — the handle already says who took the job."""
    print(json.dumps({"status": DataStage.status(_handle(args.handle)).value},
                     ensure_ascii=False))
    return 0


def cmd_cancel(args) -> int:
    handle = _handle(args.handle)
    DataStage.cancel(handle)
    print(json.dumps({"cancelled": handle.job_id}, ensure_ascii=False))
    return 0


def _run_flags() -> argparse.ArgumentParser:
    """What `submit` and `probe` both take — they are the same job, differing only in whether
    the script is contracted to produce the dataset."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--repo", required=True, help="deploy checkout root (configs/ lives there)")
    p.add_argument("--environment", default="PTB_H100_slurm")
    p.add_argument("--run-root", required=True, help="the node's run root; the recipe lives here")
    p.add_argument("--recipe", required=True, help="the script, inside run-root")
    p.add_argument("--stats", default=None,
                   help="optional JSON the recipe writes with its own per-source counts")
    p.add_argument("--shared-operators", default=None,
                   help="deployment-wide custom operator dir, prepended to the plugin path")
    # The run's resource facts, same four the other two instruments take.
    p.add_argument("--gpus", type=int, default=0,
                   help="GPUs the RECIPE uses; 0 (the default — building data is CPU work) "
                        "still gets the card this partition forces, plus the GPU-Guard "
                        "placeholder when the QOS is watched")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--qos", default=None, help="run-level QOS; env.yaml declares which exist")
    p.add_argument("--partition", default=None,
                   help="run-level partition; env.yaml names the default")
    p.add_argument("--nodelist", default=None, help="run-level node pin; overrides the env pool")
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dataos-data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    flags = _run_flags()

    s = sub.add_parser("submit", parents=[flags],
                       help="run the recipe, then check what it produced")
    s.set_defaults(fn=cmd_submit, probe=False)

    # A probe is the same job with no output contract: the script runs, nothing is appended and
    # nothing is checked. Reading a pool's schema before writing a recipe is routine, and without
    # this the only way to run python on a compute node was a hand-written sbatch that re-derived
    # the short TMPDIR, the offline-HF flags, the cache redirects and the venv activation.
    pr = sub.add_parser("probe", parents=[flags],
                        help="run a script with this stage's shell; produce nothing")
    pr.set_defaults(fn=cmd_submit, probe=True)

    t = sub.add_parser("status", help="handle -> {status}")
    t.add_argument("handle")
    t.set_defaults(fn=cmd_status)

    c = sub.add_parser("cancel", help="stop the job this handle names")
    c.add_argument("handle")
    c.set_defaults(fn=cmd_cancel)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
