from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    _add_data_commands(subparsers)
    return parser


def _add_data_commands(subparsers) -> None:
    data = subparsers.add_parser("data", help="Materialize and inspect DataOS dataset artifacts")
    data_sub = data.add_subparsers(dest="data_command")

    import_cmd = data_sub.add_parser("import", help="Run the import_dataset operator from a JSON spec")
    _add_workspace_arg(import_cmd)
    import_cmd.add_argument("--spec", required=True, help="Path to JSON spec or inline JSON object")
    import_cmd.add_argument(
        "--sandbox",
        default=None,
        help="Optional sandbox dir (defaults to <workspace>/data-import)",
    )
    import_cmd.set_defaults(func=_cmd_data_import)


def _add_workspace_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default="workspace")


def _cmd_data_import(args) -> dict[str, Any]:
    from dataos.core.sandbox import Sandbox
    from dataos.operators import build_default_registry

    spec = _load_json_arg(args.spec)
    sandbox_dir = Path(args.sandbox) if args.sandbox else Path(args.workspace) / "data-import"
    sandbox = Sandbox(sandbox_dir)
    sandbox.prepare()
    operator = build_default_registry().get("import_dataset")
    artifact = operator.run([], sandbox, **spec)
    artifact.save_meta()
    return artifact.summary()


def _load_json_arg(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {value}")
    return loaded


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
