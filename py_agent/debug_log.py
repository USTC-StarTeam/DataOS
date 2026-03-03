"""Debug log utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class DebugLogger:
    path: Path
    level: str

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def info(self, message: str) -> None:
        if self.level == "high":
            self._write("INFO", message)

    def error(self, message: str) -> None:
        if self.level == "high":
            self._write("ERROR", message)

    def event(self, obj: dict) -> None:
        self._write("EVENT", json.dumps(obj, ensure_ascii=False))

    def llm(self, message: str) -> None:
        if self.level == "high":
            self._write("LLM", message)

    def close(self) -> None:
        self._fp.close()

    def _write(self, level: str, message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._fp.write(f"{ts} [{level}] {message}\n")
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
