"""A dependency-free backend for testing the orchestration spine without a cluster."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dataos_launch.io import read_json
from dataos_launch.types import Job, LaunchPlan, Status

from ..spec import TrainSpec
from ..types import Capabilities, TrainResult


class DummyBackend:
    name = "dummy"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name,
            methods=["sft", "grpo"],
            finetuning_types=["lora", "full"],
            launch_kinds=["direct"],
            online_hook_points=["step"],
            notes="No-op backend for tests; writes a fake final_model + metrics.json.",
        )

    def validate(self, spec: TrainSpec) -> list[str]:
        errors: list[str] = []
        if not spec.base_model:
            errors.append("base_model is required")
        if not spec.run_dir:
            errors.append("run_dir is required")
        return errors

    def build(self, spec: TrainSpec) -> Job:
        run_dir = spec.run_dir
        # A tiny portable program: write final_model/config.json + metrics.json, then exit.
        # An optional overrides["exit"] forces a non-zero exit (failure-path testing).
        py = (
            "import json, os, sys\n"
            f"d = os.path.join({run_dir!r}, 'final_model')\n"
            "os.makedirs(d, exist_ok=True)\n"
            "open(os.path.join(d, 'config.json'), 'w').write('{}')\n"
            f"open(os.path.join({run_dir!r}, 'metrics.json'), 'w')."
            "write(json.dumps({'loss': 0.123, 'rows': 3}))\n"
            "sys.exit(int(os.environ.get('DUMMY_EXIT', '0')))\n"
        )
        exit_code = 0
        if isinstance(spec.overrides, dict):
            exit_code = int(spec.overrides.get("exit", 0))
        return Job(
            backend=self.name,
            job_name=spec.name or "dummy",
            command=[sys.executable, "-c", py],
            run_dir=run_dir,
            launch=LaunchPlan(kind="direct"),
            env={"DUMMY_EXIT": str(exit_code)},
        )

    def collect(self, run_dir: str) -> TrainResult:
        final = str(Path(run_dir) / "final_model")
        ok = (Path(final) / "config.json").is_file()
        metrics: dict = {}
        mpath = Path(run_dir) / "metrics.json"
        if mpath.is_file():
            try:
                loaded = read_json(mpath)
                if isinstance(loaded, dict):
                    metrics = loaded
            except (OSError, json.JSONDecodeError, ValueError):
                metrics = {}
        rows = metrics.pop("rows", None)
        status = Status.COMPLETED if ok else Status.FAILED
        return TrainResult(
            spec_id="dummy",
            status=status,
            model_path=final if ok else None,
            metrics=metrics,
            rows=rows,
            artifacts={"final_model": final},
            report_fragment={
                "stages": {"train": {"model_path": final}},
                "model": {"path": final},
                "dataset": {"rows": rows},
            },
        )
