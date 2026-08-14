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
    stage = DataStage.from_repo(Path(args.repo).resolve(), args.environment)
    spec = DataSpec(
        name=f"data_{Path(args.run_root).name}"[:64],
        run_dir=args.run_root,
        resources=Resources(
            gpus=args.gpus, cpus=args.cpus, time_h=slurm_time_s(args.time) / 3600
        ),
        recipe=args.recipe,
        stats=args.stats,
        shared_operators=args.shared_operators,
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


def cmd_status(args) -> int:
    """Polling needs no placement — the handle already says who took the job."""
    handle = Handle.model_validate(json.loads(Path(args.handle).read_text(encoding="utf-8")))
    print(json.dumps({"status": DataStage.status(handle).value}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
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
                   help="GPUs the RECIPE uses; 0 (the default — building data is CPU work) "
                        "still gets the card this partition forces, plus the GPU-Guard "
                        "placeholder when the QOS is watched")
    s.add_argument("--cpus", type=int, default=8)
    s.add_argument("--qos", default=None, help="run-level QOS; env.yaml declares which exist")
    s.add_argument("--partition", default=None,
                   help="run-level partition; env.yaml names the default")
    s.add_argument("--nodelist", default=None, help="run-level node pin; overrides the env pool")
    s.add_argument("--time", default="04:00:00")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_submit)

    t = sub.add_parser("status")
    t.add_argument("handle")
    t.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
