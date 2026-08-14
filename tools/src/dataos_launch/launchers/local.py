"""Bare-metal launcher: run the Job as a detached local subprocess."""

from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
from contextlib import suppress
from pathlib import Path

from ..io import atomic_write_json, atomic_write_text, utc_now_iso
from ..types import Handle, Job, Status
from .base import build_preamble, compose_launch_line, materialize


class LocalLauncher:
    name = "local"

    def __init__(self, **_: object) -> None:
        # accepts (and ignores) cluster kwargs for a uniform get_launcher(**cfg) signature
        pass

    def _script(self, job: Job) -> str:
        rc = f"{job.run_dir}/logs/{job.job_name}.rc"
        rc_q = shlex.quote(rc)
        lines = [
            "#!/bin/bash",
            *build_preamble(job),
            "set +e",  # capture the real exit code instead of aborting on it
            compose_launch_line(job),
            "code=$?",
            # Atomic rc so status sees the whole file.
            f"echo $code > {rc_q}.tmp && mv {rc_q}.tmp {rc_q}",
            # Reap whatever the job left in this process group. A Slurm job gets this free from
            # cgroup teardown; here nothing does it, and a leaked grandchild holds its GPU. Seen
            # for real: an eval timed out, killed the vLLM server it had spawned, and vLLM's
            # EngineCore *grandchild* survived holding 30 GiB — the next run then failed with
            # "Free memory on device (1.11/31.37 GiB)", which reads as a config error, not a leak.
            # Written inline rather than as a trap: job.pre_cmds may install its own EXIT trap
            # (the eval bridge does, to remove TMPDIR) and a second one would silently replace it.
            # `trap '' TERM` first so the group-wide signal does not take down this shell before
            # it can return the real exit code; $$ is the pgid because submit() starts a session.
            "trap '' TERM; kill -- -$$ 2>/dev/null",
            "exit $code",
        ]
        return "\n".join(lines) + "\n"

    def submit(self, job: Job) -> Handle:
        materialize(job)
        run_dir = Path(job.run_dir)
        script_path = run_dir / f"{job.job_name}.run.sh"
        atomic_write_text(script_path, self._script(job))
        rc = run_dir / "logs" / f"{job.job_name}.rc"
        if rc.exists():
            rc.unlink()
        out = run_dir / "logs" / f"{job.job_name}.out"
        fout = open(out, "wb")  # noqa: SIM115 - handed to the child; closed when it exits
        proc = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=job.run_dir,
            stdout=fout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, **job.env},
        )
        handle = Handle(
            launcher="local",
            spec_id=job.spec_id,
            run_dir=job.run_dir,
            job_name=job.job_name,
            submitted_at=utc_now_iso(),
            pid=proc.pid,
            hostname=socket.gethostname(),
            stdout_path=str(out),
        )
        atomic_write_json(run_dir / "handle.json", handle.model_dump())
        return handle

    def status(self, handle: Handle) -> Status:
        rc = Path(handle.run_dir) / "logs" / f"{handle.job_name}.rc"
        # rc is written (atomically) as the script's last act, so its presence is terminal —
        # this also avoids a zombie child reading as "alive" forever.
        if rc.is_file():
            try:
                code = int(rc.read_text(encoding="utf-8").strip() or "1")
            except ValueError:
                code = 1
            handle.exit_code = code
            return Status.COMPLETED if code == 0 else Status.FAILED
        if _pid_alive(handle.pid, handle.hostname):
            return Status.RUNNING
        return Status.FAILED  # dead with no rc -> killed/crashed before finishing

    def logs(self, handle: Handle, *, tail: int | None = None) -> str:
        return _read_tail(handle.stdout_path, tail)

    def cancel(self, handle: Handle) -> None:
        if handle.pid is None:
            return
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(handle.pid), signal.SIGTERM)


def _pid_alive(pid: int | None, hostname: str | None) -> bool:
    if pid is None:
        return False
    if hostname and hostname != socket.gethostname():
        return False  # can't introspect a pid on another host
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_tail(path: str | None, tail: int | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if tail is not None:
        return "\n".join(text.splitlines()[-tail:])
    return text
