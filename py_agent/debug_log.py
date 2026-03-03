"""Debug log utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass
class DebugLogger:
    path: Path
    level: str

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def info(self, message: str) -> None:
        if self.level == "high":
            self._write("INFO", {"message": message})

    def error(self, message: str) -> None:
        if self.level == "high":
            self._write("ERROR", {"message": message})

    def event(self, obj: dict) -> None:
        self._write("EVENT", {"event": obj})

    def llm(self, message: str) -> None:
        if self.level == "high":
            self._write("LLM", {"message": message})

    def close(self) -> None:
        self._fp.close()

    def _write(self, level: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            **payload,
        }
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()


def default_debug_log_path(cwd: str) -> Path:
    return Path(cwd).resolve() / ".py-agent" / "logs" / "debug.log"


def resolve_debug_log_path(cwd: str, log_path_arg: str | None) -> Path:
    if log_path_arg:
        path = Path(log_path_arg)
        if not path.is_absolute():
            path = Path(cwd) / path
        return path.resolve()
    return default_debug_log_path(cwd)
