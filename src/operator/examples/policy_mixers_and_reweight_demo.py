"""Run DoReMi, ODM, and Loss Reweighting demos over gsm8k + alpaca."""

from __future__ import annotations

import json

from dataos_operator import Dataset
from dataos_operator.runtime import ForwardSignal, PolicyRuntime, RuntimeContext
from dataos_operator.operators.mixer import DoremiMixer, OdmMixer
from dataos_operator.operators.reweighter import LossReweighter

from examples._policy_demo_utils import (
    add_losses,
    chunk_as_single_batch,
    ensure_output_dir,
    load_demo_datasets,
    merge_datasets,
    save_dataset,
    short_records,
    write_summary,
)


def main() -> None:
    output_dir = ensure_output_dir("policy_mixers_and_reweight_demo")
    gsm8k_dataset, alpaca_dataset = load_demo_datasets(gsm8k_limit=8, alpaca_limit=8)
    source_datasets = [gsm8k_dataset, alpaca_dataset]
    combined = merge_datasets(*source_datasets, source_name="mixture_candidates")
    batch = chunk_as_single_batch(combined)

    doremi = DoremiMixer(
        domain_key="metadata.domain",
        eta=2.0,
        epsilon=0.05,
        output_size=10,
        random_state=23,
    )
    step1_uniform = doremi.run(source_datasets)
    save_dataset(step1_uniform, output_dir / "doremi_step1_uniform.jsonl")

    doremi_runtime = PolicyRuntime([doremi], context=RuntimeContext(mode="hybrid"))
    doremi_last_action = None
    for step in range(1, 5):
        doremi_runtime.context.step = step
        current_losses = [
            1.30 - 0.06 * step if domain == "gsm8k" else 0.95 - 0.02 * step
            for domain in batch.domain_ids or []
        ]
        reference_losses = [
            1.05 - 0.03 * step if domain == "gsm8k" else 0.90 - 0.01 * step
            for domain in batch.domain_ids or []
        ]
        doremi_runtime.after_forward(
            batch,
            ForwardSignal(per_sample_losses=current_losses),
            reference_losses=reference_losses,
        )
        actions = doremi_runtime.on_update_boundary()
        doremi_last_action = actions[-1] if actions else doremi_last_action
    doremi_final_actions = doremi_runtime.finalize()
    doremi_final_action = doremi_final_actions[-1] if doremi_final_actions else doremi_last_action
    if doremi_final_action is None or doremi_final_action.domain_weights is None:
        raise RuntimeError("DoReMi demo did not produce final domain weights")
    step3_static = doremi.mix_with_weights(
        source_datasets,
        domain_weights=doremi_final_action.domain_weights,
        output_size=10,
    )
    save_dataset(step3_static, output_dir / "doremi_step3_static.jsonl")

    odm = OdmMixer(
        domain_key="metadata.domain",
        alpha=0.8,
        warmup_steps=1,
        reward_scale=8.0,
        min_exploration_rate=0.05,
        output_size=10,
        random_state=29,
    )
    odm_runtime = PolicyRuntime([odm], context=RuntimeContext(mode="online"))
    odm_last_action = None
    for step in range(1, 6):
        odm_runtime.context.step = step
        batch_losses = [
            1.25 - 0.09 * step if domain == "gsm8k" else 0.85 + 0.01 * step
            for domain in batch.domain_ids or []
        ]
        odm_runtime.after_forward(batch, ForwardSignal(per_sample_losses=batch_losses))
        actions = odm_runtime.on_update_boundary()
        odm_last_action = actions[-1] if actions else odm_last_action
    if odm_last_action is None or odm_last_action.domain_weights is None:
        raise RuntimeError("ODM demo did not produce domain weights")
    odm_mixed = odm.mix_with_weights(source_datasets, domain_weights=odm_last_action.domain_weights, output_size=10)
    save_dataset(odm_mixed, output_dir / "odm_mixed.jsonl")

    base_losses = [1.40, 1.22, 1.10, 0.98, 0.90, 0.82]
    reweight_dataset = add_losses(
        merge_datasets(
            Dataset.from_records(gsm8k_dataset[:3], metadata=dict(gsm8k_dataset.metadata)),
            Dataset.from_records(alpaca_dataset[:3], metadata=dict(alpaca_dataset.metadata)),
            source_name="reweight_batch",
        ),
        base_losses,
    )
    reweight_batch = chunk_as_single_batch(reweight_dataset)
    reweighter = LossReweighter(strategy="linupper")
    reweight_runtime = PolicyRuntime([reweighter], context=RuntimeContext(mode="online"))
    weight_actions = reweight_runtime.before_backward(
        reweight_batch,
        ForwardSignal(per_sample_losses=base_losses),
    )
    if not weight_actions or weight_actions[-1].sample_weights is None:
        raise RuntimeError("Loss reweighting demo did not produce sample weights")
    weight_action = weight_actions[-1]

    summary = {
        "datasets": {
            "gsm8k_path": str(gsm8k_dataset.metadata["path"]),
            "alpaca_path": str(alpaca_dataset.metadata["path"]),
        },
        "results": {
            "doremi": {
                "step1_records": len(step1_uniform),
                "step3_records": len(step3_static),
                "final_domain_weights": doremi_final_action.domain_weights,
                "preview": short_records(step3_static),
            },
            "odm": {
                "records": len(odm_mixed),
                "final_domain_weights": odm_last_action.domain_weights,
                "preview": short_records(odm_mixed),
            },
            "loss_reweighting": {
                "sample_weights": list(weight_action.sample_weights),
                "records": short_records(reweight_dataset, count=3),
            },
        },
    }
    write_summary(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
