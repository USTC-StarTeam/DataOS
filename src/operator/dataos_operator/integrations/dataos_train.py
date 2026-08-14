"""Bridge: expose dataos_operator's online-capable operators as dataos-train online hooks.

dataos-train discovers external online ops through the ``dataos_train.online_ops``
entry-point group — duck-typed, no import coupling (this module never imports
dataos_train). Each factory here wraps one runtime-protocol operator into train's
hook shape: ``__call__(batch: dict) -> list[float]`` per-sample factors with mean ≈ 1
(the trainer feeds live per-sample losses in ``batch["losses"]``; see
tools/docs/train/online_ops.md).
"""

from __future__ import annotations

from typing import Any

from dataos_operator.operators.reweighter.loss_reweighter import LossReweighter
from dataos_operator.operators.selector.delta_loss_selector import DeltaLossSelector
from dataos_operator.operators.selector.loss_selector import LossSelector
from dataos_operator.runtime import (
    BatchView,
    ForwardSignal,
    OperatorState,
    RuntimeContext,
)


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


class _SelectTrainHook:
    """Adapt a loss-only selector (runtime protocol) to dataos-train's ``select`` hook.

    ``observe(batch)`` forwards the trainer's per-sample-loss payload (``losses`` +
    ``row_indices``) to the operator's ``after_forward``; ``decide(ctx)`` calls
    ``on_update_boundary`` at the epoch boundary and maps the returned
    ``PolicyAction.selected_indices`` to train's ``{"indices": ...}`` decision.

    Loss-only selectors key everything on the row index (record content is merely a
    fallback loss source, which ``observe`` already supplied), so ``candidates`` is
    placeholders sized by ``ctx["n_rows"]``. Selectors needing eval/query sets or
    domain payloads (less/nice/near/tsds, odm/doremi mixers) are NOT bridgeable this
    way and stay offline/PolicyRuntime-only.
    """

    op_cls: type = LossSelector  # override in subclasses
    op_name: str = "loss_selector"

    def __init__(self, hook: str | None = None, **params: Any) -> None:
        hook_point = hook or "select"
        if hook_point != "select":
            raise ValueError(
                f"{self.op_name} supports hook point 'select' only, got {hook!r}"
            )
        self.hook_point = hook_point
        self._total_updates = params.pop("total_updates", None)
        self._op = self.op_cls(**params)
        self._state = OperatorState(name=self.op_name)
        self._ctx = RuntimeContext(mode="online")

    @property
    def name(self) -> str:
        return self.op_name

    def setup(self, ctx: Any) -> None:  # state lives on the operator instance
        return None

    def observe(self, batch: dict) -> None:
        losses = [float(x) for x in batch.get("losses") or []]
        indices = batch.get("row_indices")
        if not indices or len(indices) != len(losses):
            return  # no row identity -> cannot accumulate; stay silent (pass-through)
        self._op.after_forward(
            batch=BatchView(records=[], indices=[int(i) for i in indices]),
            signal=ForwardSignal(per_sample_losses=losses),
            state=self._state,
            context=self._ctx,
        )

    def decide(self, ctx: dict) -> dict | None:
        n = ctx.get("n_rows")
        if not n:
            return None  # dataset size unknown — leave the sampler untouched
        self._ctx.step = int(ctx.get("step", 0) or 0)
        kwargs: dict[str, Any] = {"candidates": [{} for _ in range(int(n))]}
        if self._total_updates:
            kwargs["total_updates"] = int(self._total_updates)
        action = self._op.on_update_boundary(state=self._state, context=self._ctx, **kwargs)
        if action is None or action.selected_indices is None:
            return None
        return {"indices": action.selected_indices, "domain_weights": None}


class LossSelectTrainHook(_SelectTrainHook):
    """dataos-train ``select`` hook for ``loss_selector`` (low/medium/high-loss bias)."""

    op_cls = LossSelector
    op_name = "loss_selector"


class DeltaLossSelectTrainHook(_SelectTrainHook):
    """dataos-train ``select`` hook for ``delta_loss_selector`` (delta-window curriculum)."""

    op_cls = DeltaLossSelector
    op_name = "delta_loss_selector"
