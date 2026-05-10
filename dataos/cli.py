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
    _add_operator_commands(subparsers)
    return parser


def _add_operator_commands(subparsers) -> None:
    operators = subparsers.add_parser("operators", help="Inspect built-in DataOS operators")
    operators_sub = operators.add_subparsers(dest="operators_command")

    list_cmd = operators_sub.add_parser("list", help="List operator categories, schemas, and defaults")
    list_cmd.set_defaults(func=_cmd_operators_list)


def _cmd_operators_list(args) -> list[dict[str, object]]:
    del args
    from dataos.operators import build_default_registry

    return build_default_registry().describe_all()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
