"""Registry helpers for discovering and resolving DataOS operators."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dataos.core import Operator, OperatorSpec

__all__ = [
    "DuplicateOperatorError",
    "OperatorNotFoundError",
    "OperatorRegistry",
]


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _dedupe_operator_ids(operator_ids: Sequence[str] | None) -> list[str]:
    if operator_ids is None:
        return []
    if isinstance(operator_ids, (str, bytes)):
        raise TypeError("allowed_operator_ids must be a sequence of strings")

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for operator_id in operator_ids:
        normalized = _require_text(operator_id, "operator_id")
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered_ids.append(normalized)
    return ordered_ids


class OperatorNotFoundError(KeyError):
    """Raised when the registry cannot resolve one or more operator IDs."""


class DuplicateOperatorError(ValueError):
    """Raised when multiple operators claim the same operator ID."""


class OperatorRegistry:
    """In-memory registry for operator registration, lookup, and planner discovery."""

    def __init__(self, operators: Iterable[Operator] | None = None) -> None:
        self._operators: dict[str, Operator] = {}
        if operators is not None:
            self.register_many(operators)

    def register(self, operator: Operator) -> None:
        """Register one operator instance under its declared operator ID."""

        if not isinstance(operator, Operator):
            raise TypeError("operator must implement the Operator protocol")
        if not isinstance(operator.spec, OperatorSpec):
            raise TypeError("operator.spec must be an OperatorSpec instance")

        operator_id = _require_text(operator.spec.operator_id, "operator_id")
        if operator_id in self._operators:
            raise DuplicateOperatorError(f"operator already registered: {operator_id}")
        self._operators[operator_id] = operator

    def register_many(self, operators: Iterable[Operator]) -> None:
        """Register a batch of operators in iteration order."""

        for operator in operators:
            self.register(operator)

    def get(self, operator_id: str) -> Operator:
        """Return the operator instance for a known operator ID."""

        normalized = _require_text(operator_id, "operator_id")
        try:
            return self._operators[normalized]
        except KeyError as exc:
            raise OperatorNotFoundError(f"operator not found: {normalized}") from exc

    def get_operator(self, operator_id: str) -> Operator:
        """Compatibility alias for runtime code that expects ``get_operator``."""

        return self.get(operator_id)

    def get_spec(self, operator_id: str) -> OperatorSpec:
        """Return the registered operator spec for a known operator ID."""

        return self.get(operator_id).spec

    def supports(self, operator_id: str) -> bool:
        """Return whether an operator ID is registered."""

        normalized = _require_text(operator_id, "operator_id")
        return normalized in self._operators

    def list_operator_ids(self, allowed_operator_ids: Sequence[str] | None = None) -> list[str]:
        """List registered operator IDs, optionally filtered by planner constraints."""

        if not allowed_operator_ids:
            return list(self._operators)

        operator_ids = _dedupe_operator_ids(allowed_operator_ids)
        missing = [operator_id for operator_id in operator_ids if operator_id not in self._operators]
        if missing:
            joined = ", ".join(missing)
            raise OperatorNotFoundError(f"operator not found: {joined}")
        return operator_ids

    def discover(self, allowed_operator_ids: Sequence[str] | None = None) -> list[OperatorSpec]:
        """Return operator specs for planner-side discovery and allowed-operator checks."""

        operator_ids = self.list_operator_ids(allowed_operator_ids)
        return [self.get_spec(operator_id) for operator_id in operator_ids]

    def __contains__(self, operator_id: object) -> bool:
        if not isinstance(operator_id, str):
            return False
        return operator_id.strip() in self._operators

    def __len__(self) -> int:
        return len(self._operators)
