"""Bare-metal launcher: run the Job as a detached local subprocess.

The same ``Resources`` a scheduler would enforce, enforced by the only means a bare shell has.
``time_h`` becomes ``timeout(1)``: without it a local job had no wall-clock ceiling at all, so a
hung eval on a workstation ran until someone noticed, while the identical Job under Slurm came
back TIMEOUT on schedule. Placement fields (partition, QOS, node list) mean nothing here and are
accepted and ignored, so one ``launcher_from_config`` serves both deployments.
"""

from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
from contextlib import suppress
from pathlib import Path

from ..types import Handle, Job, Status
from .base import Launcher, build_preamble, compose_launch_line

#: What ``timeout`` returns when it fired; 128+SIGKILL when the child had to be killed outright.
_TIMEOUT_RC = {124, 137}


class LocalLauncher(Launcher):
    name = "local"
    script_suffix = ".run.sh"

    def __init__(self, **_deployment: object) -> None:
        # placement is a cluster's word; accepted so get_launcher(name, **cfg) has one signature
        pass

    def render(self, job: Job) -> str:
        rc = f"{job.run_dir}/logs/{job.job_name}.rc"
        rc_q = shlex.quote(rc)
        launch = compose_launch_line(job)
        if job.resources.time_h:
            launch = f"timeout --signal=TERM {int(job.resources.time_h * 3600)} {launch}"
        lines = [
            "#!/bin/bash",
            *build_preamble(job),
            "set +e",  # capture the real exit code instead of aborting on it
            launch,
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
            # it can return the real exit code; $$ is the pgid because dispatch starts a session.
            "trap '' TERM; kill -- -$$ 2>/dev/null",
            "exit $code",
        ]
        return "\n".join(lines) + "\n"

    def dispatch(self, job: Job, script: Path) -> Handle:
        run_dir = Path(job.run_dir)
        rc = run_dir / "logs" / f"{job.job_name}.rc"
        if rc.exists():
            rc.unlink()
        out = run_dir / "logs" / f"{job.job_name}.out"
        fout = open(out, "wb")  # noqa: SIM115 - handed to the child; closed when it exits
        proc = subprocess.Popen(
            ["bash", str(script)],
            cwd=job.run_dir,
            stdout=fout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, **job.env},
        )
        # The pid IS the job id here — it is what this launcher calls the job and what it polls.
        return self.new_handle(
            job, job_id=str(proc.pid), log_path=str(out), hostname=socket.gethostname()
        )

    def status(self, handle: Handle) -> Status:
        rc = Path(handle.run_dir) / "logs" / f"{handle.job_name}.rc"
        # rc is written (atomically) as the script's last act, so its presence is terminal —
        # this also avoids a zombie child reading as "alive" forever.
        if rc.is_file():
            try:
                code = int(rc.read_text(encoding="utf-8").strip() or "1")
            except ValueError:
                code = 1
            handle.native["exit_code"] = code
            if code == 0:
                return Status.COMPLETED
            return Status.TIMEOUT if code in _TIMEOUT_RC else Status.FAILED
        if _pid_alive(handle.job_id, handle.native.get("hostname")):
            return Status.RUNNING
        return Status.FAILED  # dead with no rc -> killed/crashed before finishing

    def cancel(self, handle: Handle) -> None:
        with suppress(ProcessLookupError, PermissionError, ValueError):
            os.killpg(os.getpgid(int(handle.job_id)), signal.SIGTERM)


def _pid_alive(job_id: str, hostname: str | None) -> bool:
    if hostname and hostname != socket.gethostname():
        return False  # can't introspect a pid on another host
    try:
        os.kill(int(job_id), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True
