"""Export a policy-selected dataset bundle for LLaMA-Factory."""

from __future__ import annotations

import json

from dataos_operator.adapters import export_sharegpt_dataset, write_sft_config
from dataos_operator.operators.selector import LossSelector

from examples._policy_demo_utils import (
    add_losses,
    build_backend,
    dataset_to_messages,
    ensure_output_dir,
    load_demo_datasets,
    save_dataset,
    write_summary,
)


def main() -> None:
    output_dir = ensure_output_dir("policy_llamafactory_export_demo")
    gsm8k_dataset, alpaca_dataset = load_demo_datasets(gsm8k_limit=4, alpaca_limit=4)
    message_dataset = dataset_to_messages(gsm8k_dataset, alpaca_dataset)
    backend = build_backend(max_length=160, max_new_tokens=24)
    losses = backend.compute_losses(message_dataset.to_records())
    scored_messages = add_losses(message_dataset, losses)

    selected_dataset = LossSelector(
        sample_size=6,
        focus="high",
        random_state=31,
    ).run(scored_messages)
    save_dataset(selected_dataset, output_dir / "selected_messages.jsonl")

    dataset_info = export_sharegpt_dataset(
        selected_dataset,
        output_dir,
        dataset_name="dataos_policy_demo",
    )
    config_path = write_sft_config(
        output_dir,
        dataset_name="dataos_policy_demo",
        model_name_or_path=backend.model_name,
        extra_fields={
            "overwrite_output_dir": True,
            "logging_steps": 1,
        },
    )

    summary = {
        "datasets": {
            "gsm8k_path": str(gsm8k_dataset.metadata["path"]),
            "alpaca_path": str(alpaca_dataset.metadata["path"]),
        },
        "selected_records": len(selected_dataset),
        "dataset_info_path": str(output_dir / "dataset_info.json"),
        "config_path": str(config_path),
        "dataset_info": dataset_info,
        "first_record": dict(selected_dataset[0]) if len(selected_dataset) > 0 else None,
    }
    write_summary(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
