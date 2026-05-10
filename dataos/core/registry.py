from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataos.core.operator import Operator


@dataclass(slots=True)
class OperatorRegistry:
    _operators: dict[str, Operator] = field(default_factory=dict)

    def register(self, operator: Operator) -> None:
        if operator.name in self._operators:
            raise ValueError(f"Operator already registered: {operator.name}")
        self._operators[operator.name] = operator

    def get(self, name: str) -> Operator:
        try:
            return self._operators[name]
        except KeyError as exc:  # pragma: no cover - thin wrapper
            raise KeyError(f"Operator not found: {name}") from exc

    def list(self) -> list[Operator]:
        return [self._operators[name] for name in sorted(self._operators)]

    def describe_all(self) -> list[dict[str, Any]]:
        return [operator.describe() for operator in self.list()]
