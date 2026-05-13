from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataos.core.artifact import Artifact, EvalArtifact, ExperimentArtifact, utc_now_iso
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
    plan: dict[str, Any] | None = None
    review: dict[str, Any] | None = None

    def start(self) -> None:
        self.sandbox.prepare()
        self.status = "running"
        self.started_at = utc_now_iso()

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)

    def set_plan(self, plan: dict[str, Any]) -> None:
        self.plan = plan

    def set_review(self, review: dict[str, Any]) -> None:
        self.review = review

    def record_step(
        self,
        operator_name: str,
        input_artifact_ids: list[str],
        output_artifact_ids: list[str],
        params: dict[str, Any],
        status: str,
        error: str | None = None,
        step_type: str = "operator",
    ) -> None:
        record = {
            "step_type": step_type,
            "step_name": operator_name,
            "input_artifact_ids": input_artifact_ids,
            "output_artifact_ids": output_artifact_ids,
            "params": params,
            "status": status,
            "error": error,
        }
        if step_type == "operator":
            record["operator_name"] = operator_name
        else:
            record["runtime_name"] = operator_name
        self.steps.append(record)

    def finalize(self) -> ExperimentArtifact:
        self.status = "completed" if self.status != "failed" else self.status
        self.finished_at = utc_now_iso()
        self.score = self._infer_score()
        report = self.to_record()
        self.sandbox.write_json("report.json", report)
        artifact = ExperimentArtifact(
            artifact_id=self.exp_id,
            name=f"experiment-{self.exp_id}",
            path=self.sandbox.root,
            meta={
                "status": self.status,
                "score": self.score,
                "task_id": self.task.task_id,
                "num_steps": len(self.steps),
                "report_path": str(self.sandbox.report_path()),
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
            "experiment_root": str(self.sandbox.root),
            "artifacts": [artifact.to_record() for artifact in self.artifacts],
            "steps": self.steps,
            "status": self.status,
            "score": self.score if self.score is not None else self._infer_score(),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "plan": self.plan,
            "review": self.review,
        }

    def _infer_score(self) -> float | None:
        for artifact in reversed(self.artifacts):
            if isinstance(artifact, EvalArtifact) and artifact.meta.get("score") is not None:
                return float(artifact.meta["score"])
        return self.score
