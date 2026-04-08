from __future__ import annotations

from typing import Protocol

from datasets import Dataset

from dataos.eval.types import EvalConfig, EvalRequest, EvalResultBundle, ModelResponse


class EvalTaskSpec(Protocol):
    name: str
    version: str
    supported_request_types: set[str]

    def default_split(self) -> str: ...

    def build_requests(self, dataset: Dataset, config: EvalConfig) -> list[EvalRequest]: ...

    def score_responses(
        self,
        dataset: Dataset,
        requests: list[EvalRequest],
        responses: list[ModelResponse],
        config: EvalConfig,
    ) -> EvalResultBundle: ...
