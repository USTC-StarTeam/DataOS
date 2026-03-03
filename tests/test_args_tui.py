from __future__ import annotations

from py_agent.args import parse_args


def test_parse_tui_mode() -> None:
    args = parse_args(["--mode", "tui"])
    assert args.mode == "tui"
