"""Atomic file persistence — write to a unique temp file, then ``os.replace``.

Every receipt (plan / report / manifest / {kind}.json / project.json / eval results)
is read by concurrent pollers and sibling ``run_one`` workers; a bare ``write_text``
can be observed half-written (torn JSON), silently corrupting ``summary()`` rankings.
``os.replace`` makes the *publish* atomic on a local POSIX filesystem; the pid+uuid
temp name avoids two writers interleaving inside one temp file (threads in one pid,
pid reuse across hosts on a shared NFS mount). Last-writer-wins is still the
semantics for concurrent writers of the same path — this prevents torn reads, it is
not a lock.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

_JSON_KWARGS: dict[str, Any] = {
    "indent": 2,
    "sort_keys": True,
    "ensure_ascii": False,
    "default": str,
}


def atomic_write_text(path: Path | str, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, **_JSON_KWARGS))
