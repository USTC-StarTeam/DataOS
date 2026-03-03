"""Executable entrypoint for py-agent."""

from __future__ import annotations

import sys

from .cli import run


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
