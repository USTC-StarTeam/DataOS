from __future__ import annotations

from dataos_operator.runtime import ForwardSignal, PolicyRuntime, RuntimeContext
from dataos_operator.runtime.types import BatchView
from dataos_operator.operators.reweighter import LossReweighter


def test_loss_reweighter_outputs_normalized_sample_weights() -> None:
    batch = BatchView(
        records=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
        indices=[0, 1, 2],
        domain_ids=["gsm8k", "alpaca", "alpaca"],
    )
    runtime = PolicyRuntime([LossReweighter(strategy="linupper")], context=RuntimeContext(mode="online"))

    action = runtime.before_backward(batch, ForwardSignal(per_sample_losses=[1.4, 0.9, 0.6]))[-1]

    assert action.sample_weights is not None
    assert abs(sum(action.sample_weights) - 1.0) < 1e-6
    assert action.sample_weights[0] > action.sample_weights[-1]
