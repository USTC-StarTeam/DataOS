"""LocalLauncher: the same Job, actually run, and the ceiling a bare shell has to impose itself.

These run real subprocesses — fast ones — because the thing worth testing is that a Job written
for a cluster runs unchanged here, which no amount of string-matching on the script would show.
"""

from __future__ import annotations

import time

from dataos_launch.launchers.local import LocalLauncher
from dataos_launch.types import Job, Resources, Status


def _job(tmp_path, command, **kw) -> Job:
    base = dict(
        backend="test",
        job_name="j1",
        command=command,
        run_dir=str(tmp_path),
        resources=Resources(gpus=0, cpus=1, time_h=1.0),
    )
    base.update(kw)
    return Job(**base)


def _await(launcher, handle, timeout=30.0) -> Status:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = launcher.status(handle)
        if state.is_terminal():
            return state
        time.sleep(0.05)
    raise AssertionError(f"still {launcher.status(handle)} after {timeout}s")


def test_submit_runs_and_reports_completed(tmp_path) -> None:
    launcher = LocalLauncher()
    handle = launcher.submit(_job(tmp_path, ["echo", "hello"]))
    assert _await(launcher, handle) == Status.COMPLETED
    assert "hello" in launcher.logs(handle)
    assert handle.native["exit_code"] == 0


def test_handle_names_the_job_the_way_this_launcher_does(tmp_path) -> None:
    """``job_id`` is universal; here it is the pid. Callers used to write
    ``handle.job_id or str(handle.pid)`` to ask one question of two launchers."""
    handle = LocalLauncher().submit(_job(tmp_path, ["true"]))
    assert handle.launcher == "local" and handle.job_id.isdigit()
    assert handle.log_path and "hostname" in handle.native


def test_handle_round_trips_through_json(tmp_path) -> None:
    from dataos_launch.types import Handle

    launcher = LocalLauncher()
    handle = launcher.submit(_job(tmp_path, ["true"]))
    _await(launcher, handle)
    reloaded = Handle.model_validate_json((tmp_path / "handle.json").read_text())
    assert reloaded.job_id == handle.job_id and reloaded.log_path == handle.log_path


def test_nonzero_exit_is_failed(tmp_path) -> None:
    launcher = LocalLauncher()
    handle = launcher.submit(_job(tmp_path, ["bash", "-c", "exit 3"]))
    assert _await(launcher, handle) == Status.FAILED
    assert handle.native["exit_code"] == 3


def test_time_h_is_a_real_ceiling(tmp_path) -> None:
    """Until 2026-08-14 a local job had none: a hung eval on a workstation ran until someone
    noticed, while the identical Job under Slurm came back TIMEOUT on schedule."""
    launcher = LocalLauncher()
    handle = launcher.submit(_job(tmp_path, ["sleep", "60"], resources=Resources(
        gpus=0, cpus=1, time_h=1 / 3600)))  # one second
    assert _await(launcher, handle) == Status.TIMEOUT
