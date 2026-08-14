"""Runtime orchestrator for offline, online, and hybrid policy execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core.contracts import DataOperator, Record
from dataos_operator.dataset import Dataset
from dataos_operator.runtime.types import (
    BackwardSignal,
    BatchView,
    ForwardSignal,
    OperatorState,
    PolicyAction,
    RuntimeContext,
)


class PolicyRuntime:
    """Coordinates runtime-aware operators without assuming a specific trainer."""

    def __init__(
        self,
        operators: Sequence[DataOperator[Any, Any]],
        *,
        context: RuntimeContext | None = None,
    ) -> None:
        self.operators = list(operators)
        self.context = context or RuntimeContext()
        self.states = {
            operator.name: OperatorState(name=operator.name)
            for operator in self.operators
        }

    def prepare(self, datasets: Mapping[str, Dataset] | None = None) -> list[PolicyAction]:
        actions: list[PolicyAction] = []
        for operator in self.operators:
            state, action = operator.prepare(datasets=datasets, state=self.states[operator.name], context=self.context)
            self.states[operator.name] = state
            if action is not None:
                actions.append(action)
        return actions

    def before_batch(self, batch: BatchView, **kwargs: Any) -> list[PolicyAction]:
        actions: list[PolicyAction] = []
        for operator in self.operators:
            action = operator.before_batch(
                batch=batch,
                state=self.states[operator.name],
                context=self.context,
                **kwargs,
            )
            if action is not None:
                actions.append(action)
        return actions

    def after_forward(self, batch: BatchView, signal: ForwardSignal, **kwargs: Any) -> None:
        for operator in self.operators:
            operator.after_forward(
                batch=batch,
                signal=signal,
                state=self.states[operator.name],
                context=self.context,
                **kwargs,
            )

    def before_backward(self, batch: BatchView, signal: ForwardSignal, **kwargs: Any) -> list[PolicyAction]:
        actions: list[PolicyAction] = []
        for operator in self.operators:
            action = operator.before_backward(
                batch=batch,
                signal=signal,
                state=self.states[operator.name],
                context=self.context,
                **kwargs,
            )
            if action is not None:
                actions.append(action)
        return actions

    def after_backward(self, batch: BatchView, signal: BackwardSignal, **kwargs: Any) -> None:
        for operator in self.operators:
            operator.after_backward(
                batch=batch,
                signal=signal,
                state=self.states[operator.name],
                context=self.context,
                **kwargs,
            )

    def on_update_boundary(self, **kwargs: Any) -> list[PolicyAction]:
        actions: list[PolicyAction] = []
        for operator in self.operators:
            action = operator.on_update_boundary(
                state=self.states[operator.name],
                context=self.context,
                **kwargs,
            )
            if action is not None:
                actions.append(action)
        return actions

    def finalize(self, **kwargs: Any) -> list[PolicyAction]:
        actions: list[PolicyAction] = []
        for operator in self.operators:
            action = operator.finalize(
                state=self.states[operator.name],
                context=self.context,
                **kwargs,
            )
            if action is not None:
                actions.append(action)
        return actions

    @staticmethod
    def flatten_records(dataset: Dataset) -> list[Record]:
        return dataset.to_records()
