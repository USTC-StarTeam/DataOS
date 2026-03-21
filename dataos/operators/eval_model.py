from __future__ import annotations

import json

from dataos.core.artifact import Artifact, EvalArtifact, ModelArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox


def _find_latest_model(inputs: list[Artifact]) -> ModelArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, ModelArtifact):
            return artifact
    raise ValueError("eval_model requires at least one ModelArtifact input")


class EvalModelOperator(Operator):
    name = "eval_model"
    description = "Stub evaluate the latest model artifact and emit a score."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> EvalArtifact:
        task = kwargs.get("task")
        model_artifact = _find_latest_model(inputs)
        score = float(kwargs.get("score", 0.5))
        artifact_id = make_artifact_id("eval")
        eval_dir = sandbox.artifact_dir("eval", artifact_id)
        result = {
            "mode": kwargs.get("mode", "stub"),
            "score": score,
            "source_model_id": model_artifact.artifact_id,
            "task_id": getattr(task, "task_id", None),
        }
        (eval_dir / "results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        eval_artifact = EvalArtifact(
            artifact_id=artifact_id,
            name=f"eval-{model_artifact.artifact_id}",
            path=eval_dir,
            parent_ids=[model_artifact.artifact_id],
            producer=self.name,
            meta=result,
        )
        sandbox.log(
            "eval_model",
            f"model={model_artifact.artifact_id}\neval={eval_artifact.artifact_id}\nscore={score}\n",
        )
        return eval_artifact
