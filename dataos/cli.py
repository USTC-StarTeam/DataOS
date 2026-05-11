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
        help="Optional sandbox dir (defaults to <workspace>/sandboxes/<exp_id>)",
    )
    train.set_defaults(func=_cmd_train)


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
        "sandbox": str(result.experiment.sandbox.root),
        "initial_artifact_ids": [artifact.artifact_id for artifact in result.initial_artifacts],
        "experiment_artifact_id": result.experiment_artifact.artifact_id,
        "archived_experiment_artifact_id": result.archived_experiment_artifact.artifact_id,
        "review": result.review,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
