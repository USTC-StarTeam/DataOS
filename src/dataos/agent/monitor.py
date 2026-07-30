"""Live experiment monitor — a tiny stdlib HTTP server, embeddable or standalone.

The experiment process (``run_auto``) starts this on a port for the duration of a
run, so one process owns both the search and its live view. It also runs
standalone (``python web/server.py``) to browse the history of past runs.

It serves the single-page UI plus a read-only JSON view of every run under
``workspace/`` (each run is a directory with a ``tree.json``), and a ``/api/live``
view that the running loop publishes its current phase into. Stdlib only.

    GET /                  -> web/index.html
    GET /api/runs          -> [{id, bench, n_nodes, best_score, running, ...}]
    GET /api/run?id=<id>   -> that run's full tree.json (nodes + root)
    GET /api/live          -> {run_id, round, phase, remaining_hours, ...} or {}
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


class LiveState:
    """Thread-safe holder for the running experiment's current status.

    The loop publishes one phase at a time; ``/api/live`` serves the latest
    snapshot. Empty (``{}``) until a run publishes, and after it clears.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict = {}

    def publish(self, **fields) -> None:
        with self._lock:
            self._state.update(fields)
            self._state["updated"] = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)


# Module-level singleton: the loop writes it, the server reads it.
LIVE = LiveState()


# --- agent activity (Claude Code transcripts) -------------------------------
# Claude Code stores each session as JSONL under ~/.claude/projects/<munged-cwd>/.
# The leader's cwd is <workspace>/_agent/<run>/<role> (renamed main_agent->leader;
# both probed); the experimenter runs at the repo root. We read those transcripts
# to show what each agent is doing.
_CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MSG_TYPES = ("assistant", "user")


def _munge(path) -> str:
    """Claude Code's cwd->dirname rule: every non-alphanumeric char becomes '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def _transcript_dir(workspace: Path, run_id: str, agent: str) -> Path | None:
    if agent == "main":
        # the role dir was renamed main_agent -> leader; probe both so old runs stay visible
        for role_dir in ("leader", "main_agent"):
            direct = _CLAUDE_PROJECTS / _munge(Path(workspace) / "_agent" / run_id / role_dir)
            if direct.is_dir():
                return direct
    else:  # the experimenter runs at the repo root (one dir, newest session = live)
        direct = _CLAUDE_PROJECTS / _munge(_REPO_ROOT)
        if direct.is_dir():
            return direct
        return None
    if agent == "main":  # workspace abspath may differ — fall back to a run_id match
        hits = sorted(
            (h for pat in (f"*{_munge(run_id)}*leader", f"*{_munge(run_id)}*main-agent")
             for h in _CLAUDE_PROJECTS.glob(pat)),
            key=lambda p: p.stat().st_mtime,
        )
        return hits[-1] if hits else None
    return None


def _latest_transcript(directory: Path) -> Path | None:
    sessions = list(directory.glob("*.jsonl"))
    return max(sessions, key=lambda p: p.stat().st_mtime) if sessions else None


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "tool_result"):
                parts.append(_result_text(block.get("text") or block.get("content")))
        return "\n".join(p for p in parts if p)
    return ""


def _tool_summary(name: str, inp: dict | None) -> str:
    inp = inp or {}
    if name == "Bash":
        return inp.get("command", "")
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        return inp.get("file_path", "")
    if name == "Grep":
        return inp.get("pattern", "")
    return json.dumps(inp, ensure_ascii=False)


def transcript_events(path: Path, *, limit: int = 200, clip: int = 700) -> list[dict]:
    """Reduce a Claude Code JSONL transcript to a flat activity stream:
    assistant reasoning (text), the agent's tool calls (tool), and their
    outputs (result). Our own task prompts (large user messages) are dropped."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    events: list[dict] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") not in _MSG_TYPES:
            continue
        content = (record.get("message") or {}).get("content")
        ts = record.get("timestamp")
        if record.get("type") == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and (block.get("text") or "").strip():
                    events.append({"kind": "text", "text": block["text"].strip()[:clip], "ts": ts})
                elif block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    events.append(
                        {"kind": "tool", "name": name,
                         "text": _tool_summary(name, block.get("input"))[:clip], "ts": ts}
                    )
        elif isinstance(content, list):  # user msg: keep only tool results
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out = _result_text(block.get("content")).strip()
                    if out:
                        events.append({"kind": "result", "text": out[:clip], "ts": ts})
    for seq, event in enumerate(events):
        event["seq"] = seq
    return events[-limit:]


def activity(workspace: Path, run_id: str, agent: str) -> dict:
    directory = _transcript_dir(workspace, run_id, agent)
    session = _latest_transcript(directory) if directory else None
    if session is None:
        return {"agent": agent, "events": []}
    return {"agent": agent, "session": session.stem, "events": transcript_events(session)}


# --- sub-stage detection (data / train / eval) from the H100 run_root markers ---
# We ssh in and count the dirs the experimenter's job fills in order: data/sft built,
# training started, model merged. The cluster creds stay here (not in LiveState).
_CLUSTER: dict | None = None
_STAGE_CACHE: dict[str, tuple[float, dict]] = {}
_STAGE_TTL = 4.0  # seconds — the page polls ~every 4s; one ssh per window is plenty


def configure(cluster: dict | None) -> None:
    """Give the monitor the cluster ssh details (host/user/port/key) for /api/training."""
    global _CLUSTER
    _CLUSTER = dict(cluster) if cluster else None


def _ssh_read(remote_cmd: str, timeout: int = 8) -> str | None:
    cluster = _CLUSTER
    if not cluster or not cluster.get("host"):
        return None
    ssh = [
        "ssh", "-i", os.path.expanduser(cluster.get("key", "")),
        "-p", str(cluster.get("port", 22)),
        "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes",
        f"{cluster.get('user')}@{cluster['host']}", remote_cmd,
    ]
    try:
        done = subprocess.run(ssh, capture_output=True, text=True, timeout=timeout + 5)
        return done.stdout
    except (subprocess.SubprocessError, OSError):
        return None


def _parse_stage(text: str | None) -> dict:
    """Infer the experimenter's sub-stage (data / train / eval) from the marker dirs
    a run_root accumulates: data/sft built -> training started -> model merged."""
    if text is None or "PROBE_END" not in text:
        return {"reachable": False, "stage": None}  # truncated/failed probe
    data_n = trlog_n = model_n = 0
    for line in text.splitlines():
        if line.startswith("DATA="):
            data_n = int(line[5:] or 0)
        elif line.startswith("TRLOG="):
            trlog_n = int(line[6:] or 0)
        elif line.startswith("MODEL="):
            model_n = int(line[6:] or 0)
    if model_n > 0:
        stage = "eval"   # training + merge done -> evaluating
    elif trlog_n > 0 or data_n > 0:
        stage = "train"  # training underway, or data built and training imminent
    else:
        stage = "data"   # still building the data / preparing the job
    return {"reachable": True, "stage": stage, "trains": True}


def stage_snapshot(run_root: str) -> dict:
    now = time.time()
    cached = _STAGE_CACHE.get(run_root)
    if cached and now - cached[0] < _STAGE_TTL:
        return cached[1]
    # one ssh: count the marker dirs that mark data-built / training-started / model-merged
    probe = (
        f"RR={run_root}; "
        f"echo DATA=$(ls $RR/data/sft 2>/dev/null | wc -l); "
        f"echo TRLOG=$(ls $RR/adapter/trainer_log.jsonl $RR/final_model/trainer_log.jsonl "
        f"2>/dev/null | wc -l); "
        f"echo MODEL=$(ls $RR/final_model 2>/dev/null | wc -l); "
        f"echo PROBE_END"
    )
    data = _parse_stage(_ssh_read(probe))
    _STAGE_CACHE[run_root] = (now, data)
    return data


def _bench_of(run_id: str) -> str:
    # run_id is like "ptb-h100-qwen-gsm8k" -> the trailing token is the bench
    return run_id.rsplit("-", 1)[-1]


_HEARTBEAT_WINDOW = 45.0  # a run is "live" if its live.json beat within this many seconds


def _read_live(workspace: Path, run_id: str) -> dict:
    """The run's last persisted live snapshot (heartbeat + phase + node_run_root)."""
    try:
        return json.loads((workspace / run_id / "live.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _is_live(live: dict) -> bool:
    """True if this snapshot's heartbeat is fresh and the run isn't finished."""
    if not live or live.get("phase") == "done":
        return False
    return (time.time() - live.get("heartbeat", 0)) < _HEARTBEAT_WINDOW


def _live_status(workspace: Path, run_id: str) -> dict:
    """{running, phase} from the heartbeat — process-independent, so one monitor can
    show every run's true state (a killed run goes stale; a finished one writes done)."""
    live = _read_live(workspace, run_id)
    running = _is_live(live)
    return {"running": running, "phase": live.get("phase") if running else None}


def write_live(workspace: Path, run_id: str) -> None:
    """Persist the in-process live snapshot + a heartbeat to <run>/live.json, so any
    monitor (this run's, another run's, or the standalone) can serve its live status.
    Cluster creds are NOT in LiveState, so nothing secret is written here."""
    snap = LIVE.snapshot()
    snap["heartbeat"] = time.time()
    path = Path(workspace) / run_id / "live.json"
    try:  # atomic: write a temp then rename, so readers never see a half-written file
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snap))
        os.replace(tmp, path)
    except OSError:
        pass


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
        status = _live_status(workspace, run_id)
        runs.append(
            {
                "id": run_id,
                "bench": _bench_of(run_id),
                "n_nodes": len(nodes),
                "n_done": sum(1 for n in nodes if n.get("status") == "completed"),
                "n_failed": sum(1 for n in nodes if n.get("status") == "failed"),
                "best_score": best["score"] if best else None,
                "best_node": best["id"] if best else None,
                "running": status["running"],
                "phase": status["phase"],
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
            if path == "/api/live":
                # follow the selected run, read from its heartbeat file (one monitor,
                # many runs); fall back to the in-process snapshot if no run given.
                run = (parse_qs(route.query).get("run") or [""])[0]
                if run:
                    live = _read_live(workspace, run)
                    live["live"] = _is_live(live)
                    return self._json(live)
                return self._json(LIVE.snapshot())
            if path == "/api/activity":
                query = parse_qs(route.query)
                agent = (query.get("agent") or ["main"])[0]
                run = (query.get("run") or [""])[0]
                if agent not in ("main", "sub") or not re.fullmatch(r"[A-Za-z0-9._-]+", run):
                    return self._json({"error": "bad params"}, 400)
                return self._json(activity(workspace, run, agent))
            if path == "/api/stage":
                run = (parse_qs(route.query).get("run") or [""])[0]
                live = _read_live(workspace, run) if run else LIVE.snapshot()
                if run and not _is_live(live):
                    return self._json({"stage": None, "phase": None})  # not live -> no stage
                run_root = live.get("node_run_root")
                if live.get("phase") != "executing" or not run_root:
                    return self._json({"stage": None, "phase": live.get("phase")})
                if not live.get("node_trains", True):  # eval-only node (e.g. baseline)
                    return self._json({"stage": "eval", "trains": False})
                return self._json(stage_snapshot(run_root))
            if path == "/api/run":
                rid = (parse_qs(route.query).get("id") or [""])[0]
                if not re.fullmatch(r"[A-Za-z0-9._-]+", rid):
                    return self._json({"error": "bad id"}, 400)
                tree = _read_tree(workspace / rid / "tree.json")
                if tree is None:
                    return self._json({"error": "not found"}, 404)
                tree["meta"] = {"id": rid, "running": _live_status(workspace, rid)["running"]}
                return self._json(tree)
            return self._json({"error": "not found"}, 404)

    return Handler


def serve_forever(workspace: Path, host: str, port: int) -> None:
    """Blocking serve — used by the standalone CLI."""
    server = ThreadingHTTPServer((host, port), make_handler(workspace))
    print(f"experiment monitor: http://{host}:{port}  (workspace={workspace})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def serve_in_background(
    workspace: Path, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    """Start the monitor in a daemon thread and return the server (call
    ``shutdown()`` to stop). Used by ``run_auto`` to own a port for the run."""
    server = ThreadingHTTPServer((host, port), make_handler(workspace))
    thread = threading.Thread(target=server.serve_forever, name="dataos-monitor", daemon=True)
    thread.start()
    return server
