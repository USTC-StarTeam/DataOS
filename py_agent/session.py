"""Session persistence for CLI runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionTurn:
    messages: list[dict[str, Any]]
    events: list[dict[str, Any]]


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append_turn(self, messages: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"messages": messages, "events": events}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_turns(self) -> list[SessionTurn]:
        if not self.path.exists():
            return []
        turns: list[SessionTurn] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            turns.append(
                SessionTurn(
                    messages=[m for m in obj.get("messages", []) if isinstance(m, dict)],
                    events=[e for e in obj.get("events", []) if isinstance(e, dict)],
                )
            )
        return turns

    def load_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for turn in self.load_turns():
            messages.extend(turn.messages)
        return messages

    def load_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for turn in self.load_turns():
            events.extend(turn.events)
        return events


def default_session_path(cwd: str) -> Path:
    return Path(cwd).resolve() / ".py-agent" / "sessions" / "default.jsonl"
