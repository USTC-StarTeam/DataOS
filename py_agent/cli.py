"""CLI execution helpers."""

from __future__ import annotations

import sys

from .args import parse_args
from .config import VERSION


def run(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.version:
        print(VERSION)
        return 0
    return 0
