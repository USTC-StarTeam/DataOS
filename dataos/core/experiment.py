from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from dataos.core.artifact import Artifact, EvalArtifact, ExperimentArtifact, make_artifact_id, utc_now_iso
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate


@dataclass(slots=True)
class Experiment:
    exp_id: str
    task: TaskTemplate
    sandbox: Sandbox
    artifacts: list[Artifact] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    status: str = "created"
    score: float | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None

    def start(self) -> None:
        self.sandbox.prepare()
        self.status = "running"
        self.started_at = utc_now_iso()

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)

    def record_step(
        self,
        operator_name: str,
        input_artifact_ids: list[str],
        output_artifact_ids: list[str],
        params: dict[str, Any],
        status: str,
        error: str | None = None,
    ) -> None:
        self.steps.append(
            {
                "operator_name": operator_name,
                "input_artifact_ids": input_artifact_ids,
                "output_artifact_ids": output_artifact_ids,
                "params": params,
                "status": status,
                "error": error,
            }
        )

    def finalize(self) -> ExperimentArtifact:
        self.status = "completed" if self.status != "failed" else self.status
        self.finished_at = utc_now_iso()
        self.score = self._infer_score()
        report = self.to_record()
        self.sandbox.write_json("report.json", report)
        artifact_dir = self.sandbox.artifact_dir("experiment", self.exp_id)
        report_copy = artifact_dir / "report.json"
        report_copy.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        artifact = ExperimentArtifact(
            artifact_id=self.exp_id,
            name=f"experiment-{self.exp_id}",
            path=artifact_dir,
            meta={
                "status": self.status,
                "score": self.score,
                "task_id": self.task.task_id,
                "num_steps": len(self.steps),
                "report_path": str(report_copy),
            },
            parent_ids=[artifact.artifact_id for artifact in self.artifacts],
            producer="experiment.finalize",
        )
        artifact.save_meta()
        return artifact

    def fail(self, error: Exception | str) -> None:
        self.status = "failed"
        self.error = str(error)
        self.finished_at = utc_now_iso()
        self.sandbox.write_json("report.json", self.to_record())

    def to_record(self) -> dict[str, Any]:
        return {
            "exp_id": self.exp_id,
            "task": self.task.to_record(),
            "sandbox": str(self.sandbox.root),
            "artifacts": [artifact.to_record() for artifact in self.artifacts],
            "steps": self.steps,
            "status": self.status,
            "score": self.score,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def _infer_score(self) -> float | None:
        for artifact in reversed(self.artifacts):
            if isinstance(artifact, EvalArtifact) and artifact.meta.get("score") is not None:
                return float(artifact.meta["score"])
        return self.score


def new_experiment(task: TaskTemplate, sandbox: Sandbox) -> Experiment:
    return Experiment(exp_id=make_artifact_id("exp"), task=task, sandbox=sandbox)
