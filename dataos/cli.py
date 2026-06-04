from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    from dataos._env import load_dotenv

    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataos")
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers(dest="command")
    _add_train_commands(subparsers)
    _add_run_prepared_commands(subparsers)
    _add_operator_commands(subparsers)
    return parser


def _add_train_commands(subparsers) -> None:
    train = subparsers.add_parser("train", help="Run one DataOS task through the runtime entrypoint")
    train.add_argument("--task", required=True, help="Task template JSON path")
    train.add_argument(
        "--dataset",
        required=True,
        help="One string containing one or more dataset inputs separated by commas",
    )
    train.add_argument("--workspace", default="workspace", help="Workspace root")
    train.add_argument(
        "--sandbox",
        default=None,
        help="Optional experiment dir (defaults to <workspace>/<exp_id>)",
    )
    train.set_defaults(func=_cmd_train)


def _add_run_prepared_commands(subparsers) -> None:
    run_prepared = subparsers.add_parser(
        "run-prepared",
        help="Run fixed stub train+eval+report over an already-prepared dataset (no agent, no LLM key)",
    )
    run_prepared.add_argument("--task", required=True, help="Task template JSON path")
    run_prepared.add_argument(
        "--dataset",
        required=True,
        help="Path to the prepared training set (Alpaca/ShareGPT jsonl/json/csv or HF save_to_disk dir)",
    )
    run_prepared.add_argument("--workspace", default="workspace", help="Workspace root")
    run_prepared.add_argument(
        "--sandbox",
        default=None,
        help="Optional experiment dir (defaults to <workspace>/<exp_id>)",
    )
    run_prepared.set_defaults(func=_cmd_run_prepared)


def _add_operator_commands(subparsers) -> None:
    operators = subparsers.add_parser("operators", help="Inspect built-in DataOS operators")
    operators_sub = operators.add_subparsers(dest="operators_command")

    list_cmd = operators_sub.add_parser("list", help="List operator categories, schemas, and defaults")
    list_cmd.set_defaults(func=_cmd_operators_list)


def _cmd_operators_list(args) -> list[dict[str, object]]:
    del args
    from dataos.operators import build_default_registry

    return build_default_registry().describe_all()


def _cmd_train(args) -> dict[str, object]:
    from dataos.runtime import run_task

    datasets = [item.strip() for item in args.dataset.split(",") if item.strip()]
    result = run_task(
        task_template=args.task,
        datasets=datasets,
        workspace_dir=args.workspace,
        sandbox_root=args.sandbox,
    )
    return {
        "task_id": result.task.task_id,
        "experiment_id": result.experiment.exp_id,
        "status": result.experiment.status,
        "experiment_root": str(result.experiment.sandbox.root),
        "initial_artifact_ids": [artifact.artifact_id for artifact in result.initial_artifacts],
        "experiment_artifact_id": result.experiment_artifact.artifact_id,
        "review": result.review,
    }


def _cmd_run_prepared(args) -> dict[str, object]:
    from dataos.runtime.prepared import run_prepared

    result = run_prepared(
        task_template=args.task,
        dataset_path=args.dataset,
        workspace_dir=args.workspace,
        sandbox_root=args.sandbox,
    )
    experiment = result.experiment
    model_artifacts = [a.artifact_id for a in experiment.artifacts if a.type == "model"]
    eval_artifacts = [a.artifact_id for a in experiment.artifacts if a.type == "eval"]
    return {
        "task_id": result.task.task_id,
        "experiment_id": experiment.exp_id,
        "status": experiment.status,
        "score": experiment.score,
        "experiment_root": str(experiment.sandbox.root),
        "report_path": str(experiment.sandbox.report_path()),
        "dataset_artifact_id": result.dataset_artifact.artifact_id,
        "model_artifact_ids": model_artifacts,
        "eval_artifact_ids": eval_artifacts,
        "experiment_artifact_id": result.experiment_artifact.artifact_id,
        "num_steps": len(experiment.steps),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
