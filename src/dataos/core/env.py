"""Tiny .env loader (no external dependency).

Walks up from CWD looking for a ``.env`` file and exports any KEY=VALUE
pairs that are not already set in the process environment. Quoted values
are unquoted; lines starting with ``#`` are ignored.
"""

from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _find_dotenv(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(*, override: bool = False, start: Path | None = None) -> Path | None:
    """Load a ``.env`` file from CWD or any parent directory.

    Returns the path that was loaded, or ``None`` if no ``.env`` was found.
    Values already present in ``os.environ`` are preserved unless ``override``.
    """

    dotenv_path = _find_dotenv(start or Path.cwd())
    if dotenv_path is None:
        return None
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key.startswith("#"):
            continue
        if key in os.environ and not override:
            continue
        os.environ[key] = _strip_quotes(value.strip())
    return dotenv_path
