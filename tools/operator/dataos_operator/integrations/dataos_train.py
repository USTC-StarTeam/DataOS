"""Bridge: expose dataos_operator's online-capable operators as dataos-train online hooks.

dataos-train discovers external online ops through the ``dataos_train.online_ops``
entry-point group — duck-typed, no import coupling (this module never imports
dataos_train). Each factory here wraps one runtime-protocol operator into train's
hook shape: ``__call__(batch: dict) -> list[float]`` per-sample factors with mean ≈ 1
(the trainer feeds live per-sample losses in ``batch["losses"]``; see
tools/train/docs/online_ops.md).
"""

from __future__ import annotations

from typing import Any

from dataos_operator.operators.reweighter.loss_reweighter import LossReweighter
from dataos_operator.runtime import ForwardSignal, OperatorState, RuntimeContext


class LossReweightTrainHook:
    """Adapt :class:`LossReweighter` to dataos-train's ``reweight`` hook point.

    The operator speaks the runtime protocol (``before_backward`` -> ``PolicyAction``
    with a sum-1 softmax weight vector); the train hook contract wants per-sample
    factors with mean ≈ 1, so weights are rescaled by the batch size.
    """

    name = "loss_reweighter"

    def __init__(self, hook: str | None = None, strategy: str = "linupper",
                 delta: float = 1.0) -> None:
        hook_point = hook or "reweight"
        if hook_point != "reweight":
            raise ValueError(f"{self.name} supports hook point 'reweight' only, got {hook!r}")
        self.hook_point = hook_point
        self._op = LossReweighter(strategy=strategy, delta=delta)
        self._state = OperatorState(name=self.name)
        self._ctx = RuntimeContext(mode="online")

    def setup(self, ctx: Any) -> None:  # stateless beyond the operator's own state
        return None

    def __call__(self, batch: dict) -> list[float]:
        losses = [float(x) for x in batch["losses"]]
        action = self._op.before_backward(
            batch=None,
            signal=ForwardSignal(per_sample_losses=losses),
            state=self._state,
            context=self._ctx,
        )
        if action is None or not action.sample_weights:
            return [1.0] * len(losses)
        n = len(action.sample_weights)
        return [w * n for w in action.sample_weights]  # softmax sum=1 -> factors mean 1
