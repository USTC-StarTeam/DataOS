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
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataos")
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers(dest="command")
    catalog = subparsers.add_parser(
        "catalog",
        help="List dataos_operator operators with their constructor parameters",
    )
    catalog.set_defaults(func=_cmd_catalog)
    return parser


def _cmd_catalog(args) -> list[dict[str, object]]:
    del args
    from dataos._catalog import catalog

    return catalog()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
