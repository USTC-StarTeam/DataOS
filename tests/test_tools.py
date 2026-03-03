from __future__ import annotations

from pathlib import Path

import pytest

from py_agent.tools import create_local_tools
from py_agent.tools.base import ToolError


def test_read_write_edit_tools(tmp_path: Path) -> None:
    tools = create_local_tools(str(tmp_path))

    write_out = tools["write"].execute({"path": "a.txt", "content": "hello world"})
    assert "wrote" in write_out

    read_out = tools["read"].execute({"path": "a.txt"})
    assert read_out == "hello world"

    edit_out = tools["edit"].execute({"path": "a.txt", "old_text": "world", "new_text": "uv"})
    assert "replaced 1" in edit_out
    assert (tmp_path / "a.txt").read_text() == "hello uv"


def test_path_escape_blocked(tmp_path: Path) -> None:
    tools = create_local_tools(str(tmp_path))
    with pytest.raises(ToolError):
        tools["read"].execute({"path": "../etc/passwd"})


def test_bash_tool_executes(tmp_path: Path) -> None:
    tools = create_local_tools(str(tmp_path))
    out = tools["bash"].execute({"command": "echo hello"})
    assert "exit_code=0" in out
    assert "hello" in out
