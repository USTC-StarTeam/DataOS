"""Orchestration spine: build -> submit -> poll -> collect via DummyBackend + LocalLauncher.

The CLI test locks the JSON-in / JSON-out contract the sub-agent depends on.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dataos_launch.launchers.local import LocalLauncher
from dataos_launch.types import Status

from dataos_train.backends.dummy import DummyBackend
from dataos_train.spec import SFT, TrainSpec

_SRC = str(Path(__file__).resolve().parents[1] / "src")
_REPO = Path(__file__).resolve().parents[4]  # the DataOS repo root (configs/ lives there)


def _spec(run_dir: Path, exit_code: int = 0) -> TrainSpec:
    return TrainSpec(
        name="dummy-exp",
        backend="dummy",
        base_model="/fake/base",
        run_dir=str(run_dir),
        method=SFT(),
        overrides={"exit": exit_code},
    )


def _poll(launcher: LocalLauncher, handle, timeout: float = 30.0) -> Status:
    st = launcher.status(handle)
    t0 = time.monotonic()
    while not st.is_terminal() and time.monotonic() - t0 < timeout:
        time.sleep(0.05)
        st = launcher.status(handle)
    return st


def test_dummy_local_success(tmp_path: Path) -> None:
    spec = _spec(tmp_path / "ok")
    backend, launcher = DummyBackend(), LocalLauncher()
    handle = launcher.submit(backend.build(spec))
    assert (tmp_path / "ok" / "handle.json").is_file()
    assert _poll(launcher, handle) == Status.COMPLETED
    result = backend.collect(handle.run_dir)
    assert result.status == Status.COMPLETED
    assert result.model_path and Path(result.model_path, "config.json").is_file()
    assert result.rows == 3
    assert result.report_fragment["model"]["path"].endswith("final_model")


def test_dummy_local_failure(tmp_path: Path) -> None:
    spec = _spec(tmp_path / "fail", exit_code=3)
    backend, launcher = DummyBackend(), LocalLauncher()
    handle = launcher.submit(backend.build(spec))
    assert _poll(launcher, handle) == Status.FAILED


def test_cli_run_dummy(tmp_path: Path) -> None:
    run_dir = tmp_path / "cli"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec(run_dir).model_dump(), default=str), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": _SRC}
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "dataos_train.cli",
            "run",
            str(spec_path),
            # The environment names the launcher (LOCAL declares `launcher: local`); there is no
            # --launcher flag any more, because naming one was a fourth place placement could be
            # decided and the one furthest from the config.
            "--repo",
            str(_REPO),
            "--environment",
            "example-local",
            "--poll",
            "0.05",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["status"] == "COMPLETED"
    assert payload["model_path"].endswith("final_model")
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "job.json").is_file()


def test_cli_capabilities() -> None:
    env = {**os.environ, "PYTHONPATH": _SRC}
    out = subprocess.run(
        [sys.executable, "-m", "dataos_train.cli", "capabilities", "dummy"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    caps = json.loads(out.stdout.strip().splitlines()[-1])
    assert caps["name"] == "dummy"
    assert "sft" in caps["methods"]


def test_cli_schema() -> None:
    env = {**os.environ, "PYTHONPATH": _SRC}
    out = subprocess.run(
        [sys.executable, "-m", "dataos_train.cli", "schema", "--backend", "llamafactory"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["backend"]["name"] == "llamafactory"
    for method in ("sft", "dpo", "ppo", "kto", "orpo"):
        assert method in payload["methods"]
    assert "learning_rate" in payload["methods"]["sft"]["properties"]
    assert "runtime_owned_fields" in payload["agent_contract"]

    out = subprocess.run(
        [sys.executable, "-m", "dataos_train.cli", "schema", "--method", "grpo"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    grpo = json.loads(out.stdout.strip().splitlines()[-1])
    assert "rollout" in grpo["properties"]
    assert "reward" in grpo["properties"]

    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "dataos_train.cli",
            "schema",
            "--backend",
            "verl",
            "--method",
            "sft",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode != 0
    err = json.loads(out.stdout.strip().splitlines()[-1])
    assert "does not support method" in err["error"]
