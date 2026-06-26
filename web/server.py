"""Tiny stdlib HTTP server for the live experiment-tree monitor.

Serves a single static page plus a read-only JSON view of every run under
``workspace/`` (each run is a directory with a ``tree.json``). No external
deps, no build step — run it and open the page; it polls these endpoints.

    GET /                  -> web/index.html
    GET /api/runs          -> [{id, bench, n_nodes, best_score, running, ...}]
    GET /api/run?id=<id>   -> that run's full tree.json (nodes + root)

Usage:  python web/server.py [--port 8765] [--workspace <dir>]
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parent


def _bench_of(run_id: str) -> str:
    # run_id is like "ptb-h100-qwen-gsm8k" -> the trailing token is the bench
    return run_id.rsplit("-", 1)[-1]


def _is_running(workspace: Path, run_id: str) -> bool:
    """A run is done once its log records a STOP line; otherwise treat as live."""
    log = workspace / "_logs" / run_id / "run.log"
    if not log.exists():
        return True
    try:
        tail = log.read_text(errors="replace")[-4000:]
    except OSError:
        return True
    return "STOP:" not in tail


def _read_tree(tree_path: Path) -> dict | None:
    try:
        return json.loads(tree_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def list_runs(workspace: Path) -> list[dict]:
    runs: list[dict] = []
    for tree_path in sorted(workspace.glob("*/tree.json")):
        run_id = tree_path.parent.name
        if run_id.startswith("_"):
            continue
        tree = _read_tree(tree_path)
        if tree is None:
            continue
        nodes = tree.get("nodes", [])
        scored = [n for n in nodes if isinstance(n.get("score"), (int, float))]
        best = max(scored, key=lambda n: n["score"], default=None)
        runs.append(
            {
                "id": run_id,
                "bench": _bench_of(run_id),
                "n_nodes": len(nodes),
                "n_done": sum(1 for n in nodes if n.get("status") == "completed"),
                "n_failed": sum(1 for n in nodes if n.get("status") == "failed"),
                "best_score": best["score"] if best else None,
                "best_node": best["id"] if best else None,
                "running": _is_running(workspace, run_id),
                "updated": tree_path.stat().st_mtime,
            }
        )
    # live runs first, then most-recently-updated
    runs.sort(key=lambda r: (not r["running"], -r["updated"]))
    return runs


def make_handler(workspace: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):  # quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json; charset=utf-8", code)

        def do_GET(self):  # noqa: N802
            route = urlparse(self.path)
            path = route.path
            if path in ("/", "/index.html"):
                html = (WEB_DIR / "index.html").read_bytes()
                return self._send(html, "text/html; charset=utf-8")
            if path == "/api/runs":
                return self._json(list_runs(workspace))
            if path == "/api/run":
                rid = (parse_qs(route.query).get("id") or [""])[0]
                if not re.fullmatch(r"[A-Za-z0-9._-]+", rid):
                    return self._json({"error": "bad id"}, 400)
                tree = _read_tree(workspace / rid / "tree.json")
                if tree is None:
                    return self._json({"error": "not found"}, 404)
                tree["meta"] = {"id": rid, "running": _is_running(workspace, rid)}
                return self._json(tree)
            return self._json({"error": "not found"}, 404)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--workspace", default=str(REPO_ROOT / "workspace"))
    args = ap.parse_args()
    workspace = Path(args.workspace).resolve()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(workspace))
    print(f"experiment monitor: http://{args.host}:{args.port}  (workspace={workspace})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
