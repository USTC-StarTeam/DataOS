"""Placeholder iterative controller implementation for the MVP."""

from __future__ import annotations

from dataos.controller.base import ControllerDecision, ControllerRoundContext

__all__ = ["IterativeController"]


class IterativeController:
    """Reserved MVP controller slot for bounded round-based optimization."""

    controller_id = "iterative"

    def decide(self, context: ControllerRoundContext) -> ControllerDecision:  # pragma: no cover
        raise NotImplementedError(
            "IterativeController is a placeholder boundary. Concrete control logic lands in a later task."
        )
