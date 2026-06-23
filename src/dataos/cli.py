from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    from dataos.core.env import load_dotenv

    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = _cmd_run(args)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataos",
        description="Run the PostTrainBench agent loop from one JSON config.",
    )
    parser.add_argument("config", help="PTB run config JSON path")
    return parser


def _cmd_run(args: argparse.Namespace) -> dict[str, object]:
    from dataos.agent.loop import run_auto
    from dataos.agent.run_config import load_auto_run_config

    return run_auto(load_auto_run_config(args.config))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
