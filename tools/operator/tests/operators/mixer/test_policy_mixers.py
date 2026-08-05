from __future__ import annotations

from dataos_operator import Dataset
from dataos_operator.runtime import ForwardSignal, PolicyRuntime, RuntimeContext
from dataos_operator.runtime.types import BatchView
from dataos_operator.operators.mixer import DoremiMixer, OdmMixer


def test_doremi_runtime_updates_and_finalizes_domain_weights() -> None:
    dataset = Dataset.from_records(
        [
            {"id": "a", "metadata": {"domain": "gsm8k"}},
            {"id": "b", "metadata": {"domain": "gsm8k"}},
            {"id": "c", "metadata": {"domain": "alpaca"}},
            {"id": "d", "metadata": {"domain": "alpaca"}},
        ]
    )
    batch = BatchView(
        records=dataset.to_records(),
        indices=[0, 1, 2, 3],
        domain_ids=["gsm8k", "gsm8k", "alpaca", "alpaca"],
    )
    mixer = DoremiMixer(domain_key="metadata.domain", eta=2.0, epsilon=0.05, output_size=4, random_state=5)
    runtime = PolicyRuntime([mixer], context=RuntimeContext(mode="hybrid", step=1))

    runtime.after_forward(
        batch,
        ForwardSignal(per_sample_losses=[1.3, 1.2, 0.7, 0.8]),
        reference_losses=[1.0, 1.0, 0.7, 0.75],
    )
    action = runtime.on_update_boundary()[-1]
    final_action = runtime.finalize()[-1]

    assert action.domain_weights is not None
    assert action.domain_weights["gsm8k"] > action.domain_weights["alpaca"]
    assert final_action.domain_weights is not None
    assert set(final_action.domain_weights) == {"gsm8k", "alpaca"}


def test_odm_runtime_shifts_weight_toward_high_loss_domain() -> None:
    dataset = Dataset.from_records(
        [
            {"id": "a", "metadata": {"domain": "gsm8k"}},
            {"id": "b", "metadata": {"domain": "alpaca"}},
        ]
    )
    batch = BatchView(records=dataset.to_records(), indices=[0, 1], domain_ids=["gsm8k", "alpaca"])
    mixer = OdmMixer(domain_key="metadata.domain", alpha=0.8, warmup_steps=0, reward_scale=5.0, random_state=7)
    runtime = PolicyRuntime([mixer], context=RuntimeContext(mode="online", step=2))

    runtime.after_forward(batch, ForwardSignal(per_sample_losses=[1.4, 0.8]))
    action = runtime.on_update_boundary()[-1]

    assert action.domain_weights is not None
    assert action.domain_weights["gsm8k"] > action.domain_weights["alpaca"]
