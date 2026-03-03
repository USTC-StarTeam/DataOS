"""Argument parsing for the minimal CLI agent."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .config import APP_NAME


@dataclass(frozen=True)
class CliArgs:
    version: bool
    print_mode: bool
    mode: str
    model: str
    no_session: bool
    session: str | None
    continue_session: bool
    messages: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="Minimal Python coding agent",
        add_help=True,
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("-p", "--print", dest="print_mode", action="store_true", help="Run one-shot print mode")
    parser.add_argument(
        "--mode",
        choices=["text", "json"],
        default="text",
        help="Output mode for print mode",
    )
    parser.add_argument("--model", default="glm-5", help="Model ID (default: glm-5)")
    parser.add_argument("--no-session", action="store_true", help="Disable session persistence")
    parser.add_argument("--session", default=None, help="Session JSONL file path")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Continue previous session")
    parser.add_argument("messages", nargs="*", help="Prompt text")
    return parser


def parse_args(argv: list[str]) -> CliArgs:
    ns = build_parser().parse_args(argv)
    return CliArgs(
        version=bool(ns.version),
        print_mode=bool(ns.print_mode),
        mode=str(ns.mode),
        model=str(ns.model),
        no_session=bool(ns.no_session),
        session=ns.session,
        continue_session=bool(ns.continue_session),
        messages=list(ns.messages),
    )
