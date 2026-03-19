"""Placeholder offline evaluator for the MVP boundary skeleton."""

from __future__ import annotations

from dataos.eval.base import EvaluationRequest, EvaluatorSpec

__all__ = ["OfflineProxyEvaluator"]


class OfflineProxyEvaluator:
    """Reserved MVP evaluator slot for cheap offline proxy scoring."""

    spec = EvaluatorSpec(
        evaluator_id="offline_proxy",
        name="Offline Proxy Evaluator",
        metrics=["quality_proxy"],
        description="Phase 0 placeholder for offline candidate scoring.",
    )

    def evaluate(self, request: EvaluationRequest):  # pragma: no cover - placeholder only.
        raise NotImplementedError(
            "OfflineProxyEvaluator is a placeholder boundary. Concrete scoring lands in a later task."
        )
