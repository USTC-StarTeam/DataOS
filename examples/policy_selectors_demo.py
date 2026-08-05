"""Run paper-inspired selector demos over gsm8k + alpaca with the unified policy runtime."""

from __future__ import annotations

import json

from dataos_operator import Dataset
from dataos_operator.runtime import ForwardSignal, PolicyRuntime, RuntimeContext
from dataos_operator.operators.selector import (
    DeltaLossSelector,
    LessSelector,
    LossSelector,
    NearSelector,
    NiceSelector,
    TsdsSelector,
)

from examples._policy_demo_utils import (
    add_embeddings,
    add_losses,
    build_backend,
    chunk_as_single_batch,
    dataset_from_indices,
    ensure_output_dir,
    load_demo_datasets,
    merge_datasets,
    runtime_loss_schedule,
    save_dataset,
    short_records,
    write_summary,
)


def main() -> None:
    output_dir = ensure_output_dir("policy_selectors_demo")
    gsm8k_dataset, alpaca_dataset = load_demo_datasets(gsm8k_limit=4, alpaca_limit=4)
    candidate_dataset = merge_datasets(gsm8k_dataset, alpaca_dataset, source_name="selector_candidates")
    backend = build_backend(max_length=160, max_new_tokens=24)

    base_losses = backend.compute_losses(candidate_dataset.to_records())
    scored_candidates = add_losses(candidate_dataset, base_losses, initial_loss_key="metadata.runtime.initial_loss")
    domains = [str(record["metadata"]["domain"]) for record in scored_candidates]

    loss_selected = LossSelector(
        sample_size=4,
        focus="high",
        random_state=17,
    ).run(scored_candidates)
    save_dataset(loss_selected, output_dir / "loss_selected.jsonl")

    delta_selector = DeltaLossSelector(
        sample_size=4,
        window_fraction=0.4,
        random_state=17,
    )
    delta_runtime = PolicyRuntime(
        [delta_selector],
        context=RuntimeContext(mode="online", metadata={"total_updates": 4}),
    )
    batch = chunk_as_single_batch(scored_candidates)
    delta_action = None
    for step in range(1, 5):
        delta_runtime.context.step = step
        step_losses = runtime_loss_schedule(base_losses, domains, step)
        delta_runtime.after_forward(batch, ForwardSignal(per_sample_losses=step_losses))
        actions = delta_runtime.on_update_boundary(candidates=scored_candidates.to_records(), total_updates=4)
        delta_action = actions[-1] if actions else None
    if delta_action is None or delta_action.selected_indices is None:
        raise RuntimeError("DeltaLossSelector did not produce a selection action")
    delta_selected = dataset_from_indices(
        scored_candidates,
        delta_action.selected_indices,
        label="delta_loss",
    )
    save_dataset(delta_selected, output_dir / "delta_loss_selected.jsonl")

    embedding_candidates = add_embeddings(
        candidate_dataset,
        backend.compute_embeddings(candidate_dataset.to_records()),
    )
    embedding_queries = add_embeddings(
        gsm8k_dataset,
        backend.compute_embeddings(gsm8k_dataset.to_records()),
    )
    near_selected = NearSelector(
        embedding_queries,
        sample_size=4,
        max_neighbors=4,
    ).run(embedding_candidates)
    tsds_selected = TsdsSelector(
        embedding_queries,
        sample_size=4,
        max_neighbors=4,
        kde_neighbors=3,
        sigma=0.9,
        alpha=0.5,
        budget=4.0,
        random_state=17,
    ).run(embedding_candidates)
    save_dataset(near_selected, output_dir / "near_selected.jsonl")
    save_dataset(tsds_selected, output_dir / "tsds_selected.jsonl")

    model_candidates = merge_datasets(
        Dataset.from_records(gsm8k_dataset[:1], metadata=dict(gsm8k_dataset.metadata)),
        Dataset.from_records(alpaca_dataset[:1], metadata=dict(alpaca_dataset.metadata)),
        source_name="model_select_candidates",
    )
    model_eval = Dataset.from_records(gsm8k_dataset[:1], metadata=dict(gsm8k_dataset.metadata))
    model_context = RuntimeContext(mode="hybrid", metadata={"backend": backend})

    less_runtime = PolicyRuntime(
        [LessSelector(model_eval, sample_size=1, projection_dim=64, eval_sample_size=1)],
        context=model_context,
    )
    less_runtime.prepare()
    less_action = less_runtime.on_update_boundary(
        candidates=model_candidates.to_records(),
        backend=backend,
    )[-1]
    if less_action.selected_indices is None:
        raise RuntimeError("LessSelector did not produce a selection action")
    less_selected = dataset_from_indices(model_candidates, less_action.selected_indices, label="less")
    save_dataset(less_selected, output_dir / "less_selected.jsonl")

    nice_runtime = PolicyRuntime(
        [
            NiceSelector(
                model_eval,
                sample_size=1,
                projection_dim=64,
                eval_sample_size=1,
                reward_samples=1,
                temperature=0.8,
            )
        ],
        context=model_context,
    )
    nice_runtime.prepare()
    nice_action = nice_runtime.on_update_boundary(
        candidates=model_candidates.to_records(),
        backend=backend,
    )[-1]
    if nice_action.selected_indices is None:
        raise RuntimeError("NiceSelector did not produce a selection action")
    nice_selected = dataset_from_indices(model_candidates, nice_action.selected_indices, label="nice")
    save_dataset(nice_selected, output_dir / "nice_selected.jsonl")

    summary = {
        "model_name": backend.model_name,
        "datasets": {
            "gsm8k_path": str(gsm8k_dataset.metadata["path"]),
            "alpaca_path": str(alpaca_dataset.metadata["path"]),
        },
        "results": {
            "loss": {
                "records": len(loss_selected),
                "preview": short_records(loss_selected),
            },
            "delta_loss": {
                "records": len(delta_selected),
                "selected_indices": list(delta_action.selected_indices),
                "preview": short_records(delta_selected),
            },
            "near": {
                "records": len(near_selected),
                "preview": short_records(near_selected),
            },
            "tsds": {
                "records": len(tsds_selected),
                "preview": short_records(tsds_selected),
            },
            "less": {
                "records": len(less_selected),
                "selected_indices": list(less_action.selected_indices),
                "preview": short_records(less_selected),
            },
            "nice": {
                "records": len(nice_selected),
                "selected_indices": list(nice_action.selected_indices),
                "preview": short_records(nice_selected),
            },
        },
    }
    write_summary(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
