"""Registry helpers for strategy-like objects such as planners and controllers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "DuplicateStrategyError",
    "StrategyNotFoundError",
    "StrategyRegistry",
]


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _dedupe_strategy_ids(strategy_ids: Sequence[str] | None) -> list[str]:
    if strategy_ids is None:
        return []
    if isinstance(strategy_ids, (str, bytes)):
        raise TypeError("strategy_ids must be a sequence of strings")

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for strategy_id in strategy_ids:
        normalized = _require_text(strategy_id, "strategy_id")
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered_ids.append(normalized)
    return ordered_ids


class StrategyNotFoundError(KeyError):
    """Raised when the registry cannot resolve one or more strategy IDs."""


class DuplicateStrategyError(ValueError):
    """Raised when multiple strategies claim the same strategy ID."""


class StrategyRegistry:
    """Generic in-memory registry for strategy objects keyed by string IDs."""

    def __init__(
        self,
        strategies: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    ) -> None:
        self._strategies: dict[str, Any] = {}
        if strategies is not None:
            items = strategies.items() if isinstance(strategies, Mapping) else strategies
            self.register_many(items)

    def register(self, strategy_id: str, strategy: Any) -> None:
        """Register one strategy object under an explicit ID."""

        normalized = _require_text(strategy_id, "strategy_id")
        if normalized in self._strategies:
            raise DuplicateStrategyError(f"strategy already registered: {normalized}")
        self._strategies[normalized] = strategy

    def register_many(self, strategies: Iterable[tuple[str, Any]]) -> None:
        """Register a batch of named strategies in iteration order."""

        for strategy_id, strategy in strategies:
            self.register(strategy_id, strategy)

    def get(self, strategy_id: str) -> Any:
        """Return the strategy object for a known strategy ID."""

        normalized = _require_text(strategy_id, "strategy_id")
        try:
            return self._strategies[normalized]
        except KeyError as exc:
            raise StrategyNotFoundError(f"strategy not found: {normalized}") from exc

    def supports(self, strategy_id: str) -> bool:
        """Return whether a strategy ID is registered."""

        normalized = _require_text(strategy_id, "strategy_id")
        return normalized in self._strategies

    def list_strategy_ids(self, strategy_ids: Sequence[str] | None = None) -> list[str]:
        """List registered strategy IDs, optionally filtered by explicit IDs."""

        if not strategy_ids:
            return list(self._strategies)

        normalized_ids = _dedupe_strategy_ids(strategy_ids)
        missing = [strategy_id for strategy_id in normalized_ids if strategy_id not in self._strategies]
        if missing:
            joined = ", ".join(missing)
            raise StrategyNotFoundError(f"strategy not found: {joined}")
        return normalized_ids

    def __contains__(self, strategy_id: object) -> bool:
        if not isinstance(strategy_id, str):
            return False
        return strategy_id.strip() in self._strategies

    def __len__(self) -> int:
        return len(self._strategies)
