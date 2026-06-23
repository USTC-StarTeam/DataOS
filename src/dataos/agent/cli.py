"""CLI entry: ``python -m dataos.agent run ...`` (or the ``dataos-auto`` script).

Builds an :class:`AutoRunConfig` from flags and drives :func:`dataos.agent.loop.run_auto`.
The stop policy needs at least a target (``--target``) or a budget cap
(``--max-rounds`` / ``--max-wall-clock-s``) so the run always terminates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PTB_WALL_CLOCK_S, AutoRunConfig, Budget, Criterion


def _load_json_arg(value: str | None, *, label: str) -> tuple[dict, str | None]:
    """Load a JSON object from a file path or inline JSON string."""
    if not value:
        return {}, None
    raw = value[1:] if value.startswith("@") else value
    path = Path(raw)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    else:
        data = json.loads(value)
        source = None
    if not isinstance(data, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return data, source


def _build_config(args: argparse.Namespace) -> AutoRunConfig:
    eval_cfg = json.loads(args.eval_cfg) if args.eval_cfg else None
    slurm, _ = _load_json_arg(args.slurm, label="--slurm")
    runtime_config, runtime_config_path = _load_json_arg(
        args.runtime_config, label="--runtime-config"
    )
    gpus = [g for g in (args.gpus.split(",") if args.gpus else []) if g != ""]
    # PTB's single stop condition is wall-clock; if the user gave no cap, default to 10h so the
    # loop always terminates (a bare --max-rounds is recorded but does NOT stop the loop).
    wall = args.max_wall_clock_s if args.max_wall_clock_s is not None else PTB_WALL_CLOCK_S
    return AutoRunConfig(
        project_id=args.project,
        instruct=args.instruct or "",
        goal=args.goal or "",
        scenario=args.scenario,
        criterion=Criterion(metric=args.metric, threshold=args.target, direction=args.direction),
        budget=Budget(max_rounds=args.max_rounds, max_wall_clock_s=wall),
        gpus=gpus,
        workspace=args.workspace,
        poll_interval_s=args.poll_interval_s,
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        claude_code_model=args.claude_code_model,
        max_steps=args.max_steps,
        data_source=args.data_source,
        data_limit=args.data_limit,
        eval_cfg=eval_cfg,
        stub=args.stub,
        ptb_profile=args.ptb_profile,
        target_bench=args.target_bench,
        train_executor=args.train_executor,
        slurm=slurm,
        runtime_config=runtime_config,
        runtime_config_path=runtime_config_path,
        h100_base_model=args.h100_base_model,
        base_model_id=args.base_model_id,
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
    run.add_argument(
        "--backend", default="claude_code", choices=["claude_code", "claude", "openai", "stub"],
        help="main-agent backend (default claude_code = local Claude Code subscription)",
    )
    run.add_argument("--model", default="mimo-v2.5-pro", help="model id for claude/openai backends")
    run.add_argument(
        "--base-url", default="https://token-plan-cn.xiaomimimo.com/anthropic", dest="base_url"
    )
    run.add_argument(
        "--claude-code-model", dest="claude_code_model", default="claude-sonnet-4-6",
        help="claude_code backend model (DataOS runs default to Sonnet)",
    )
    run.add_argument("--max-steps", type=int, default=12, dest="max_steps")

    # execution target — train locally, or ship train+gen to the H100 Slurm cluster
    run.add_argument(
        "--train-executor", dest="train_executor", default="local",
        choices=["local", "subagent"],
        help="local = deterministic run_one (stub/dev); subagent = an agentic sub-agent "
             "implements each experiment (operators + H100 train/eval), the real PTB path",
    )
    run.add_argument(
        "--target-bench", dest="target_bench", default=None,
        help="single PTB bench (per-(base,bench) optimization); omit for the 7-bench suite",
    )
    run.add_argument(
        "--h100-base-model", dest="h100_base_model", default=None,
        help="H100 -Base model path for the subagent executor (e.g. "
             "/data/hcgu/hf_cache/hub/Qwen3-1.7B-Base)",
    )
    run.add_argument("--base-model-id", dest="base_model_id", default=None, help="logical HF id")
    run.add_argument(
        "--slurm", default=None,
        help="Legacy JSON overrides for H100 connection defaults; prefer --runtime-config",
    )
    run.add_argument(
        "--runtime-config", dest="runtime_config", default=None,
        help="JSON file or inline JSON with cluster paths/envs, e.g. configs/runtimes/h100.json",
    )

    # data / eval
    run.add_argument("--data-source", dest="data_source", help="hf://... or local path (prepare-once)")
    run.add_argument("--data-limit", type=int, dest="data_limit")
    run.add_argument("--eval-cfg", dest="eval_cfg", help="frozen eval config as a JSON string")
    run.add_argument("--stub", action="store_true", help="tiny data + stub train/eval (no GPU/HF)")
    run.add_argument(
        "--ptb-profile", default="formal", choices=["formal", "smoke"], dest="ptb_profile",
        help="PostTrainBench limit profile (eval is fixed to PTB on this branch)",
    )

    args = ap.parse_args(argv)

    # Stop policy always exists: --max-wall-clock-s defaults to PTB's 10h in _build_config, so the
    # loop is guaranteed to terminate even with no explicit cap. (--target/--max-rounds are recorded
    # but do NOT stop the loop on this PTB-fixed branch.)

    from .loop import run_auto

    result = run_auto(_build_config(args))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
