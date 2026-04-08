from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dataos.core.artifact import DatasetArtifact, ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.eval.types import EvalConfig, EvalRunOutput


@dataclass(slots=True)
class EvalRunContext:
    dataset_artifact: DatasetArtifact | None
    model_artifact: ModelArtifact
    sandbox: Sandbox
    task_template: Any
    config: EvalConfig


class EvalProvider(Protocol):
    name: str

    def run(self, context: EvalRunContext) -> EvalRunOutput: ...
