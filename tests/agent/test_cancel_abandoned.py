"""A node the runtime gives up on must not leave its jobs running.

Nothing cancelled anything before this: `grep cancel src/dataos/agent/` matched two comments.
Every Launcher had implemented `cancel()` since the split and no caller existed, so an abandoned
node's work ran on — two trainings shared one card on the workstation, and on the cluster a dead
node's data build held `debug_normal`'s single job slot for seven minutes while the next node
waited behind it.
"""

from __future__ import annotations

import json
import time
import types
from pathlib import Path

from dataos_launch.launchers.local import LocalLauncher
from dataos_launch.types import Job, Resources

from dataos.agent import runtime

_SFT_ENV = "/zssd/home/tangjiahui/2026/DataOS-Space/envs/dataos_sft/.venv"


def _cfg():
    """A LOCAL-shaped deployment: no host, so the exec host is this machine."""
    return types.SimpleNamespace(
        runtime_config={"cluster": {}, "paths": {"sft_env": _SFT_ENV}}, project_id="t"
    )


def _submit(run_dir: Path, name: str, command: list[str]):
    return LocalLauncher().submit(Job(
        backend="test", job_name=name, command=command, run_dir=str(run_dir),
        resources=Resources(gpus=0, cpus=1, time_h=1),
    ))


def _group_size(pid: str) -> int:
    """Processes left in the job's process group — the launcher starts each job in its own."""
    import subprocess

    out = subprocess.run(["ps", "-eo", "pgid", "--no-headers"], capture_output=True, text=True)
    return sum(1 for line in out.stdout.split() if line.strip() == str(pid))


def _await_rc(run_dir: Path, name: str, timeout: float = 20.0) -> None:
    rc = run_dir / "logs" / f"{name}.rc"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not rc.is_file():
        time.sleep(0.05)


def test_only_the_live_job_is_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_run_root", lambda cfg, sid, name: str(tmp_path))

    done = _submit(tmp_path, "finished", ["true"])
    _await_rc(tmp_path, "finished")
    live = _submit(tmp_path, "live", ["sleep", "600"])
    # a run_dir holds one handle per job PLUS `handle.json` pointing at the last one, so the
    # sweep must dedupe by job id or it would report the same cancellation twice
    assert len(list(tmp_path.glob("*handle.json"))) == 3

    cancelled = runtime.cancel_live_jobs(_cfg(), "study", "node")
    assert cancelled == [live.job_id]          # the finished one is not claimed
    assert done.job_id not in cancelled

    # ...and it really stopped, rather than merely being reported. Asserted on the process
    # GROUP, not on `status()`: pytest is the submitting process here and never reaps, so the
    # killed wrapper lingers as a zombie and `os.kill(pid, 0)` still succeeds. In production the
    # submitter is a `dataos-*` CLI that exits at once, the child is reparented and reaped, and
    # status reads FAILED — the zombie is an artifact of a parent that outlives its job.
    time.sleep(2)
    assert _group_size(live.job_id) == 0

    # idempotent: nothing is in flight now, so nothing is claimed
    assert runtime.cancel_live_jobs(_cfg(), "study", "node") == []


def test_a_record_that_is_not_handle_shaped_is_skipped(tmp_path, monkeypatch):
    """dataos-eval writes its own `<name>.eval_handle.json` wrapping the launcher's handle under
    a `launch` key. It is not a Handle, and the launcher's own file covers the same job."""
    monkeypatch.setattr(runtime, "_run_root", lambda cfg, sid, name: str(tmp_path))
    (tmp_path / "x.eval_handle.json").write_text(json.dumps({"launch": {"job_id": "1"}}))
    (tmp_path / "junk.handle.json").write_text("not json at all")

    assert runtime.cancel_live_jobs(_cfg(), "study", "node") == []
