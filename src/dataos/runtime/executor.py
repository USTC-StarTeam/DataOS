"""Fan out a study's experiments as run_one.py subprocesses.

This is the part that replaces "the agent hand-launches nohup": the loop owns N child
processes via ``subprocess.Popen`` (no shell ``&``, so nothing gets swallowed by a
harness backgrounding the whole command), pins one GPU per process, limits concurrency
with the GPU pool as a semaphore, and treats each experiment's ``report.json`` as the
authoritative status (atomic writes in the library guarantee no torn reads).

Local-only for the MVP; ``SSHRunner`` is a reserved seam (see bottom).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dataos.core.study import _slug


def _report_path(workspace: str, project_id: str, study_id: str, name: str) -> Path:
    return Path(workspace) / project_id / study_id / "experiments" / _slug(name) / "report.json"


def _read_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _spec_name(spec) -> str:
    return spec.name if hasattr(spec, "name") else spec["name"]


def fan_out(
    *,
    project_id: str,
    study_id: str,
    workspace: str,
    specs: list,
    data_dir: str,
    gpus: list,
    run_one: str = "scripts/run_one.py",
    python: str | None = None,
    cwd: str | None = None,
    poll_interval_s: float = 10.0,
    timeout_s: float | None = None,
    skip_completed: bool = True,
    split: str = "train",
    limit: int | None = None,
    extra_env: dict | None = None,
    sleep=time.sleep,
) -> dict[str, dict]:
    """Run every spec to terminal state; return ``{spec_name: report_dict}``.

    Concurrency = len(gpus) (each process gets one ``CUDA_VISIBLE_DEVICES``). Pass an
    empty list for a single un-pinned slot (stub/CPU). ``report.json`` is the source of
    truth; a process that exits without a completed report is recorded as failed.
    """
    python = python or sys.executable
    cwd = os.path.abspath(cwd or os.getcwd())
    workspace = os.path.abspath(workspace)
    data_dir = os.path.abspath(data_dir)
    run_one_abs = run_one if os.path.isabs(run_one) else os.path.join(cwd, run_one)

    names = [_spec_name(s) for s in specs]
    reports: dict[str, dict] = {}
    queue: list[str] = []
    for name in names:
        if skip_completed:
            r = _read_report(_report_path(workspace, project_id, study_id, name))
            if r is not None and r.get("status") == "completed":
                reports[name] = r
                continue
        queue.append(name)

    gpu_pool: list = list(gpus) if gpus else [None]  # None → no CUDA pin, single slot
    logs_dir = Path(workspace) / project_id / study_id / "_auto_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple] = {}  # name -> (Popen, gpu, logfile, started_monotonic)

    def launch(name: str) -> None:
        gpu = gpu_pool.pop()
        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            python, run_one_abs,
            "--project", project_id, "--study", study_id, "--spec", name,
            "--data", data_dir, "--workspace", workspace, "--split", split,
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        logf = open(logs_dir / f"{_slug(name)}.log", "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        running[name] = (proc, gpu, logf, time.monotonic())

    while queue and gpu_pool:
        launch(queue.pop(0))

    while running:
        finished: list[str] = []
        for name, (proc, gpu, logf, started) in list(running.items()):
            rc = proc.poll()
            timed_out = timeout_s is not None and (time.monotonic() - started) > timeout_s
            if rc is None and not timed_out:
                continue
            if rc is None and timed_out:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
                rc = proc.returncode
            logf.close()
            gpu_pool.append(gpu)
            report = _read_report(_report_path(workspace, project_id, study_id, name))
            if report is None:
                report = {
                    "name": name,
                    "status": "failed",
                    "score": None,
                    "error": f"no report.json (exit rc={rc}, timeout={timed_out})",
                }
            elif report.get("status") != "completed":
                report = {**report, "status": "failed",
                          "error": f"process exited rc={rc}, timeout={timed_out}"}
            reports[name] = report
            finished.append(name)
        for name in finished:
            running.pop(name)
            if queue and gpu_pool:
                launch(queue.pop(0))
        if running:
            sleep(poll_interval_s)

    return reports


class SSHRunner:  # pragma: no cover - reserved seam, not implemented in MVP
    """Reserved: run ``run_one.py`` on a remote host and read ``report.json`` back.

    The MVP is local-only (``fan_out``). A remote runner would prefix the command with
    ``ssh <host>`` and pull report.json via scp/sshfs — see docs/runbooks/remote.md.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("SSHRunner is a reserved seam; the MVP fans out locally.")
