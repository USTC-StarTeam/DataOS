"""CLI entry: ``python -m dataos_auto run ...`` (or the ``dataos-auto`` script).

Builds an :class:`AutoRunConfig` from flags and drives :func:`dataos_auto.loop.run_auto`.
The stop policy needs at least a target (``--target``) or a budget cap
(``--max-rounds`` / ``--max-wall-clock-s``) so the run always terminates.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import AutoRunConfig, Budget, Criterion


def _build_config(args: argparse.Namespace) -> AutoRunConfig:
    eval_cfg = json.loads(args.eval_cfg) if args.eval_cfg else None
    gpus = [g for g in (args.gpus.split(",") if args.gpus else []) if g != ""]
    return AutoRunConfig(
        project_id=args.project,
        instruct=args.instruct or "",
        goal=args.goal or "",
        scenario=args.scenario,
        criterion=Criterion(metric=args.metric, threshold=args.target, direction=args.direction),
        budget=Budget(max_rounds=args.max_rounds, max_wall_clock_s=args.max_wall_clock_s),
        gpus=gpus,
        workspace=args.workspace,
        poll_interval_s=args.poll_interval_s,
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        max_steps=args.max_steps,
        data_source=args.data_source,
        data_limit=args.data_limit,
        eval_cfg=eval_cfg,
        stub=args.stub,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataos-auto", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the autonomous loop")

    run.add_argument("--project", required=True, help="project id (open-or-create; resumable)")
    run.add_argument("--goal", help="one-line goal")
    run.add_argument("--instruct", help="full user instruction")
    run.add_argument("--scenario", default="generic")

    # stop policy (three hard conditions)
    run.add_argument("--target", type=float, help="bench score target (criterion threshold)")
    run.add_argument("--metric", default="score", help="'score' (headline/macro) or a bench name")
    run.add_argument("--direction", default="max", choices=["max", "min"])
    run.add_argument("--max-rounds", type=int, dest="max_rounds")
    run.add_argument("--max-wall-clock-s", type=float, dest="max_wall_clock_s")

    # execution
    run.add_argument("--workspace", default="workspace")
    run.add_argument("--gpus", default="", help="comma-separated device ids, e.g. '4,5,6'")
    run.add_argument("--poll-interval-s", type=float, default=10.0, dest="poll_interval_s")

    # LLM backend
    run.add_argument("--backend", default="claude", choices=["claude", "stub"])
    run.add_argument("--model", default="mimo-v2.5-pro")
    run.add_argument(
        "--base-url", default="https://token-plan-cn.xiaomimimo.com/anthropic", dest="base_url"
    )
    run.add_argument("--max-steps", type=int, default=12, dest="max_steps")

    # data / eval
    run.add_argument("--data-source", dest="data_source", help="hf://... or local path (prepare-once)")
    run.add_argument("--data-limit", type=int, dest="data_limit")
    run.add_argument("--eval-cfg", dest="eval_cfg", help="frozen eval config as a JSON string")
    run.add_argument("--stub", action="store_true", help="tiny data + stub train/eval (no GPU/HF)")

    args = ap.parse_args(argv)

    if args.target is None and args.max_rounds is None and args.max_wall_clock_s is None:
        ap.error("need a stop policy: pass --target and/or --max-rounds / --max-wall-clock-s")

    from .loop import run_auto

    result = run_auto(_build_config(args))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
