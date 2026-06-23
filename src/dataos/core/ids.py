"""Shared id + timestamp helpers (leaf module; stdlib only)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _make_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"
