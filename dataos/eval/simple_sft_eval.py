"""Placeholder SFT evaluator that runs after runtime pipeline execution."""

from __future__ import annotations

from dataos.eval.base import EvaluationRequest, EvaluatorSpec

__all__ = ["SimpleSFTEvaluator"]


class SimpleSFTEvaluator:
    """Reserved MVP evaluator slot for post-runtime SFT dataset feedback."""

    spec = EvaluatorSpec(
        evaluator_id="simple_sft_eval",
        name="Simple SFT Evaluator",
        metrics=["eval.win_rate", "quality.pass_rate"],
        description="Phase 0/T0 placeholder for evaluating candidate SFT datasets after runtime.",
    )

    def evaluate(self, request: EvaluationRequest):  # pragma: no cover - placeholder only.
        raise NotImplementedError(
            "SimpleSFTEvaluator is a placeholder boundary. Concrete offline SFT scoring lands in T1."
        )
