"""Atomic IO + id/time helpers.

Shapes mirror ``dataos.core.io`` / ``dataos.core.ids`` so artifacts written by
``dataos-train`` (spec.json / handle.json / result.json) are torn-read safe — pollers
read them concurrently with writers.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp without microseconds."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def make_id(prefix: str) -> str:
    """``<prefix>-<YYYYMMDDHHMMSSffffff>-<uuid8>`` (matches dataos.core.ids)."""
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}-{ts}-{uuid4().hex[:8]}"


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` atomically via a temp file + ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    return atomic_write_text(path, text)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
