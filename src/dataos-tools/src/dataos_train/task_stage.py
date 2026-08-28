"""Config-selected frozen training workflows.

This is the train front door for tasks whose recipe belongs to an external harness.  DataOS
schedules one job and records its handle; the configured executable performs every training
operation inside that job.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataos_launch import Job
from dataos_launch.spec import Spec
from dataos_launch.stage import Stage
from dataos_launch.task_config import (
    load_stage_contract,
    tool_provider,
    validate_provider_pin,
)


class TaskTrainSpec(Spec):
    stage = "train"

    task_config: str
    suite_dir: str
    submission_path: str


class TaskTrainStage(Stage):
    """Build an external training workflow from ``train/task.yaml``."""

    name = "train"

    def __init__(self, env: dict, *, repo: Path, task_config: str, contract: dict) -> None:
        super().__init__(env)
        self.repo = repo
        self.task_config = task_config
        self.contract = contract

    @classmethod
    def from_repo(
        cls, repo: str | Path, environment: str, task_config: str, **_: object
    ) -> TaskTrainStage:
        repo = Path(repo)
        _, contract = load_stage_contract(repo, task_config, "train")
        if contract.get("driver") != "external_cli":
            raise ValueError(
                f"task {task_config!r} train driver is {contract.get('driver')!r}; "
                "task-submit requires external_cli"
            )
        return cls(
            cls.load_env(repo, environment),
            repo=repo,
            task_config=task_config,
            contract=contract,
        )

    def _tool(self) -> tuple[dict, Path]:
        name = str(self.contract["tool"])
        provider = tool_provider(self.env, name)
        return provider, Path(str(provider["path"])) / "bin" / name

    def validate(self, spec: TaskTrainSpec) -> list[str]:
        errors = []
        if spec.task_config != self.task_config:
            errors.append(
                f"spec task_config {spec.task_config!r} does not match {self.task_config!r}"
            )
        if not Path(spec.suite_dir, "suite_state.json").is_file():
            errors.append(f"suite is not initialized: {spec.suite_dir}/suite_state.json")
        if not Path(spec.submission_path).is_dir():
            errors.append(f"submission dataset does not exist: {spec.submission_path}")
        try:
            tool = str(self.contract["tool"])
            errors.extend(validate_provider_pin(tool_provider(self.env, tool), tool))
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
        workflow = self.contract.get("workflow")
        if not isinstance(workflow, list) or not workflow:
            errors.append("train/task.yaml workflow must be a non-empty list")
        return errors

    def build(self, spec: TaskTrainSpec) -> Job:
        provider, executable = self._tool()
        values = {
            "suite_dir": spec.suite_dir,
            "submission_path": spec.submission_path,
            "run_dir": spec.run_dir,
        }
        steps = []
        for raw in self.contract["workflow"]:
            args = [str(value).format_map(values) for value in raw.get("args") or []]
            steps.append(
                {
                    "name": str(raw["name"]),
                    "args": args,
                    "accept_status": [str(value) for value in raw.get("accept_status") or []],
                }
            )
        manifest = {
            "executable": str(executable),
            "cwd": str(provider.get("root") or Path(executable).parents[2]),
            "steps": steps,
            "result": self.contract.get("result") or {},
            "output": str(Path(spec.run_dir) / "external_train_result.json"),
        }
        runner = Path(tool_provider(self.env, "dataos-train")["path"]) / "bin" / "python"
        return Job(
            backend="external_cli",
            job_name=f"train_{spec.name}",
            command=[
                str(runner),
                "-m",
                "dataos_train.external_runner",
                str(Path(spec.run_dir) / "external_train_manifest.json"),
            ],
            env={str(k): str(v) for k, v in (provider.get("env") or {}).items()},
            run_dir=spec.run_dir,
            resources=spec.resources,
            assets={"external_train_manifest.json": json.dumps(manifest, indent=2)},
        )
