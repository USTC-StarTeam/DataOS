from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from dataos.core.artifact import EvalArtifact, ModelArtifact, ResultArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
class EvalArtifactStore:
    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def save_requests(
        self,
        records: list[dict[str, Any]],
        parent_ids: list[str],
        meta: dict[str, Any],
    ) -> ResultArtifact:
        return self._save_result_artifact(
            name=f"eval-requests-{parent_ids[-1]}",
            kind="eval_requests",
            filename="requests.jsonl",
            records=records,
            parent_ids=parent_ids,
            meta=meta,
        )

    def save_responses(
        self,
        records: list[dict[str, Any]],
        parent_ids: list[str],
        meta: dict[str, Any],
    ) -> ResultArtifact:
        return self._save_result_artifact(
            name=f"eval-responses-{parent_ids[-1]}",
            kind="eval_responses",
            filename="responses.jsonl",
            records=records,
            parent_ids=parent_ids,
            meta=meta,
        )

    def save_samples(
        self,
        records: list[dict[str, Any]],
        parent_ids: list[str],
        meta: dict[str, Any],
    ) -> ResultArtifact:
        return self._save_result_artifact(
            name=f"eval-samples-{parent_ids[-1]}",
            kind="eval_samples",
            filename="samples.jsonl",
            records=records,
            parent_ids=parent_ids,
            meta=meta,
        )

    def save_eval_artifact(
        self,
        *,
        model_artifact: ModelArtifact,
        parent_ids: list[str],
        summary: dict[str, Any],
        sample_records: list[dict[str, Any]],
    ) -> EvalArtifact:
        artifact_id = make_artifact_id("eval")
        eval_dir = self.sandbox.artifact_dir("eval", artifact_id)
        (eval_dir / "results.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        if sample_records:
            self._write_jsonl(eval_dir / "samples.jsonl", sample_records)
            self._write_jsonl(eval_dir / "predictions.jsonl", sample_records)
        eval_artifact = EvalArtifact(
            artifact_id=artifact_id,
            name=f"eval-{model_artifact.artifact_id}",
            path=eval_dir,
            parent_ids=parent_ids,
            producer="eval_model",
            meta=summary,
        )
        return eval_artifact

    def _save_result_artifact(
        self,
        *,
        name: str,
        kind: str,
        filename: str,
        records: list[dict[str, Any]],
        parent_ids: list[str],
        meta: dict[str, Any],
    ) -> ResultArtifact:
        artifact_id = make_artifact_id("result")
        result_dir = self.sandbox.artifact_dir("result", artifact_id)
        self._write_jsonl(result_dir / filename, records)
        artifact = ResultArtifact(
            artifact_id=artifact_id,
            name=name,
            path=result_dir,
            parent_ids=parent_ids,
            producer="eval_model",
            meta={
                "kind": kind,
                **meta,
            },
        )
        return artifact

    def _write_jsonl(self, path: Path, records: Iterable[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
