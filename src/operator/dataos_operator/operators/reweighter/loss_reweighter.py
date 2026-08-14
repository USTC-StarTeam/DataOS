"""Loss-based sample reweighting for trainer-aware optimization."""

from __future__ import annotations

from typing import Any

from dataos_operator.core import Record, ReweighterOperator
from dataos_operator.runtime import OperatorState, PolicyAction, RuntimeContext


class LossReweighter(ReweighterOperator):
    """Reweight samples according to their relative loss values.

    Implementation: normalize the losses to a symmetric range and map them to raw weights by
    ``strategy`` (``linupper``/``uniform``/``quadratic``/``extremes``, scaled by ``delta``), then
    softmax into a normalized weight vector. The offline ``reweight_record`` reads ``loss_key`` and
    writes a per-record weight to ``weight_key``; the runtime ``before_backward`` hook returns the
    batch's per-sample weights as a PolicyAction (to scale the loss before backprop).
    Dependency: pure-rule operator, no external dependencies; online use needs per-sample losses
    fed through the PolicyRuntime.
    """

    def __init__(
        self,
        *,
        loss_key: str = "metadata.runtime.loss",
        weight_key: str = "metadata.runtime.weight",
        strategy: str = "linupper",
        delta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            loss_key=loss_key,
            weight_key=weight_key,
            strategy=strategy,
            delta=delta,
            **kwargs,
        )
        if strategy not in {"linupper", "uniform", "quadratic", "extremes"}:
            raise ValueError("strategy must be one of: linupper, uniform, quadratic, extremes")

        self.loss_key = loss_key
        self.weight_key = weight_key
        self.strategy = strategy
        self.delta = float(delta)

    def reweight_record(self, record: Record) -> Record:
        loss = self.get_field(record, self.loss_key, None)
        if loss is None:
            return dict(record)
        weight = self._weights([float(loss)])[0]
        return self.set_field(record, self.weight_key, weight)

    def before_backward(
        self,
        *,
        batch: Any,
        signal: Any,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        del batch, state, context, kwargs
        if signal.per_sample_losses is None:
            return None
        weights = self._weights([float(loss) for loss in signal.per_sample_losses])
        return PolicyAction(
            sample_weights=weights,
            metadata={"reweighter": self.name, "strategy": self.strategy},
        )

    def _weights(self, losses: list[float]) -> list[float]:
        if not losses:
            return []
        minimum = min(losses)
        maximum = max(losses)
        denominator = max(maximum - minimum, 1e-6)
        normalized = [
            2.0 * self.delta * loss / denominator - self.delta * (maximum + minimum) / denominator
            for loss in losses
        ]
        if self.strategy == "linupper":
            raw = [min(value + self.delta, self.delta) for value in normalized]
        elif self.strategy == "uniform":
            raw = normalized
        elif self.strategy == "quadratic":
            raw = [1 - (value ** 2) / (self.delta ** 2) for value in normalized]
        else:
            raw = [abs(value) for value in normalized]
        shifted = [pow(2.718281828, value - max(raw)) for value in raw]
        total = sum(shifted)
        return [value / total for value in shifted]
