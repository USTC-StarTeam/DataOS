"""Standalone monitor CLI — browse the history of every run under ``workspace/``.

The serving logic lives in ``src/dataos/agent/monitor.py`` (one implementation,
shared with the embedded monitor that ``run_auto`` starts during a live run).
This wrapper loads that module by path so it stays pure-stdlib: no install, no
dataos package import, no external deps.

    python web/server.py [--port 8765] [--host 0.0.0.0] [--workspace <dir>]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_MONITOR_PY = REPO_ROOT / "src" / "dataos" / "agent" / "monitor.py"


def _load_monitor():
    spec = importlib.util.spec_from_file_location("dataos_monitor", _MONITOR_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--workspace", default=str(REPO_ROOT / "workspace"))
    args = ap.parse_args()
    _load_monitor().serve_forever(Path(args.workspace).resolve(), args.host, args.port)


if __name__ == "__main__":
    main()
