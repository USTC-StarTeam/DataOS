from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest


def _sanitize_node_name(nodeid: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", nodeid)


@pytest.fixture
def test_workspace(request: pytest.FixtureRequest) -> Path:
    """Provide a fixed workspace under tests/.tmp and clean it after each test."""
    root = Path("tests/.tmp").resolve()
    case_dir = root / _sanitize_node_name(request.node.nodeid)

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        yield case_dir
    finally:
        if case_dir.exists():
            shutil.rmtree(case_dir)
