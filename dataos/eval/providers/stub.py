from __future__ import annotations

from dataos.eval.providers.base import EvalRunContext
from dataos.eval.types import EvalResultBundle, EvalRunOutput


class StubEvalProvider:
    name = "stub"

    def run(self, context: EvalRunContext) -> EvalRunOutput:
        score = float(context.config.provider_config.get("score", 0.5))
        return EvalRunOutput(
            provider=self.name,
            task_name=None,
            task_version=None,
            adapter_name=None,
            request_type=None,
            dataset_split=None,
            summary=EvalResultBundle(
                metric_name="stub_score",
                score=score,
                summary={},
                samples=[],
            ),
        )
