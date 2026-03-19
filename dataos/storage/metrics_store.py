"""Metric storage interfaces shared by runtime and eval."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dataos.core.eval import EvalResult

__all__ = ["MetricsStore"]


@runtime_checkable
class MetricsStore(Protocol):
    """Boundary contract for persisting step- and eval-level metrics."""

    def save_step_metrics(self, run_id: str, step_id: str, metrics: dict[str, Any]) -> None:
        """Persist structured metrics for one pipeline step."""

    def save_eval_result(self, result: EvalResult) -> None:
        """Persist one structured evaluation result."""
