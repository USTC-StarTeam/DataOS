"""Compatibility module for ``python -m dataos.agent.cli``.

The user-facing CLI is ``dataos <run-config.json>``. This module intentionally
shares that implementation instead of exposing a second flag-based interface.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from dataos.cli import main as dataos_main

    return dataos_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
