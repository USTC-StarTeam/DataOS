from __future__ import annotations

from typing import Protocol

from dataos.core.artifact import ModelArtifact
from dataos.eval.types import EvalConfig, EvalRequest, ModelResponse


class EvalModelAdapter(Protocol):
    name: str

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: EvalConfig,
    ) -> list[ModelResponse]: ...
