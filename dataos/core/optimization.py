"""Optimization loop state shared by controller and storage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from dataos.core.eval import EvalResult


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return deepcopy(value)


@dataclass(slots=True)
class OptimizationState:
    """Structured state for one optimization round."""

    round_id: str
    candidate_configs: list[dict[str, Any]] = field(default_factory=list)
    tried_configs: list[dict[str, Any]] = field(default_factory=list)
    best_config: dict[str, Any] | None = None
    best_result: EvalResult | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.round_id:
            raise ValueError("round_id must not be empty")
        self.candidate_configs = [deepcopy(config) for config in self.candidate_configs]
        self.tried_configs = [deepcopy(config) for config in self.tried_configs]
        self.best_config = _copy_mapping(self.best_config)
        if self.best_result is not None and not isinstance(self.best_result, EvalResult):
            self.best_result = EvalResult(**self.best_result)
        self.history = [deepcopy(entry) for entry in self.history]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the optimization state."""

        return asdict(self)


__all__ = ["OptimizationState"]
