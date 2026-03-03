from __future__ import annotations

from py_agent.args import parse_args


def test_parse_mcp_flags() -> None:
    args = parse_args([
        "-p",
        "--mcp-config",
        "./mcp.json",
        "--mcp-server",
        "demo",
        "--mcp-server",
        "calc",
        "hello",
    ])
    assert args.mcp_config == "./mcp.json"
    assert args.mcp_servers == ["demo", "calc"]
    assert args.messages == ["hello"]
