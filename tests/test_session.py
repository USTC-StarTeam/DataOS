from __future__ import annotations

from pathlib import Path

from py_agent.session import SessionStore, default_session_path


def test_session_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    store = SessionStore(path)

    store.append_turn(
        messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        events=[{"type": "user", "content": "hello"}, {"type": "assistant", "content": "hi"}],
    )
    store.append_turn(
        messages=[{"role": "user", "content": "next"}, {"role": "assistant", "content": "done"}],
        events=[{"type": "user", "content": "next"}, {"type": "assistant", "content": "done"}],
    )

    assert store.load_messages() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "done"},
    ]
    assert [e["type"] for e in store.load_events()] == ["user", "assistant", "user", "assistant"]


def test_default_session_path_is_under_dot_py_agent(tmp_path: Path) -> None:
    p = default_session_path(str(tmp_path))
    assert ".py-agent" in str(p)
    assert p.name == "default.jsonl"
