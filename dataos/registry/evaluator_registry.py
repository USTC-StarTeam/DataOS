"""Registry helpers for discovering and resolving DataOS evaluators."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dataos.eval.base import Evaluator, EvaluatorSpec

__all__ = [
    "DuplicateEvaluatorError",
    "EvaluatorNotFoundError",
    "EvaluatorRegistry",
]


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _dedupe_evaluator_ids(evaluator_ids: Sequence[str] | None) -> list[str]:
    if evaluator_ids is None:
        return []
    if isinstance(evaluator_ids, (str, bytes)):
        raise TypeError("evaluator_ids must be a sequence of strings")

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for evaluator_id in evaluator_ids:
        normalized = _require_text(evaluator_id, "evaluator_id")
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered_ids.append(normalized)
    return ordered_ids


class EvaluatorNotFoundError(KeyError):
    """Raised when the registry cannot resolve one or more evaluator IDs."""


class DuplicateEvaluatorError(ValueError):
    """Raised when multiple evaluators claim the same evaluator ID."""


class EvaluatorRegistry:
    """In-memory registry for evaluator lookup and planner/controller discovery."""

    def __init__(self, evaluators: Iterable[Evaluator] | None = None) -> None:
        self._evaluators: dict[str, Evaluator] = {}
        if evaluators is not None:
            self.register_many(evaluators)

    def register(self, evaluator: Evaluator) -> None:
        """Register one evaluator instance under its declared evaluator ID."""

        if not isinstance(evaluator, Evaluator):
            raise TypeError("evaluator must implement the Evaluator protocol")
        if not isinstance(evaluator.spec, EvaluatorSpec):
            raise TypeError("evaluator.spec must be an EvaluatorSpec instance")

        evaluator_id = _require_text(evaluator.spec.evaluator_id, "evaluator_id")
        if evaluator_id in self._evaluators:
            raise DuplicateEvaluatorError(f"evaluator already registered: {evaluator_id}")
        self._evaluators[evaluator_id] = evaluator

    def register_many(self, evaluators: Iterable[Evaluator]) -> None:
        """Register a batch of evaluators in iteration order."""

        for evaluator in evaluators:
            self.register(evaluator)

    def get(self, evaluator_id: str) -> Evaluator:
        """Return the evaluator instance for a known evaluator ID."""

        normalized = _require_text(evaluator_id, "evaluator_id")
        try:
            return self._evaluators[normalized]
        except KeyError as exc:
            raise EvaluatorNotFoundError(f"evaluator not found: {normalized}") from exc

    def supports(self, evaluator_id: str) -> bool:
        """Return whether an evaluator ID is registered."""

        normalized = _require_text(evaluator_id, "evaluator_id")
        return normalized in self._evaluators

    def list_evaluator_ids(self, evaluator_ids: Sequence[str] | None = None) -> list[str]:
        """List registered evaluator IDs, optionally filtered by explicit IDs."""

        if not evaluator_ids:
            return list(self._evaluators)

        normalized_ids = _dedupe_evaluator_ids(evaluator_ids)
        missing = [evaluator_id for evaluator_id in normalized_ids if evaluator_id not in self._evaluators]
        if missing:
            joined = ", ".join(missing)
            raise EvaluatorNotFoundError(f"evaluator not found: {joined}")
        return normalized_ids

    def discover(self, evaluator_ids: Sequence[str] | None = None) -> list[EvaluatorSpec]:
        """Return evaluator specs for controller-side discovery and validation."""

        return [self._evaluators[evaluator_id].spec for evaluator_id in self.list_evaluator_ids(evaluator_ids)]

    def __contains__(self, evaluator_id: object) -> bool:
        if not isinstance(evaluator_id, str):
            return False
        return evaluator_id.strip() in self._evaluators

    def __len__(self) -> int:
        return len(self._evaluators)
