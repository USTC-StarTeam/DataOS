"""Argument parsing for the minimal CLI agent."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .config import APP_NAME


@dataclass(frozen=True)
class CliArgs:
    help: bool
    version: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="Minimal Python coding agent (PR1 skeleton)",
        add_help=True,
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    return parser


def parse_args(argv: list[str]) -> CliArgs:
    ns = build_parser().parse_args(argv)
    return CliArgs(help=False, version=bool(ns.version))
