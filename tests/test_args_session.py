from __future__ import annotations

from py_agent.args import parse_args


def test_parse_session_flags() -> None:
    args = parse_args(["-p", "--session", "s.jsonl", "--continue", "hello"])
    assert args.print_mode is True
    assert args.session == "s.jsonl"
    assert args.continue_session is True
    assert args.no_session is False
    assert args.messages == ["hello"]


def test_parse_no_session_flag() -> None:
    args = parse_args(["-p", "--no-session", "hello"])
    assert args.no_session is True
    assert args.session is None
