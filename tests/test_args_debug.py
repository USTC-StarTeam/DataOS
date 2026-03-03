from __future__ import annotations

from py_agent.args import parse_args


def test_parse_debug_flags() -> None:
    args = parse_args(["--debug", "high", "--log-path", "./logs/run.log", "--version"])
    assert args.debug == "high"
    assert args.log_path == "./logs/run.log"
    assert args.version is True


def test_parse_debug_default_level_low() -> None:
    args = parse_args(["--debug", "--version"])
    assert args.debug == "low"
