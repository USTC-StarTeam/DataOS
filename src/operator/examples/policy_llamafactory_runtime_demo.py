"""Run a tiny vendored LLaMA-Factory SFT job with live DataOS policy control."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dataos_operator.adapters import export_sharegpt_dataset
from dataos_operator.core import set_field

from examples._policy_demo_utils import (
    DEFAULT_MODEL,
    build_backend,
    dataset_to_messages,
    ensure_output_dir,
    load_demo_datasets,
    write_summary,
)

ROOT = Path(__file__).resolve().parents[1]
LLAMAFACTORY_SRC = ROOT / "LLaMA-Factory" / "src"
if str(LLAMAFACTORY_SRC) not in sys.path:
    sys.path.insert(0, str(LLAMAFACTORY_SRC))
os.environ.setdefault("LLAMAFACTORY_ALLOW_SINGLE_PROCESS", "1")
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

from llamafactory.train.tuner import run_exp  # noqa: E402


def main() -> None:
    output_dir = ensure_output_dir("policy_llamafactory_runtime_demo")
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    gsm8k_dataset, alpaca_dataset = load_demo_datasets(gsm8k_limit=2, alpaca_limit=2)
    message_dataset = dataset_to_messages(gsm8k_dataset, alpaca_dataset)
    eval_dataset = dataset_to_messages(gsm8k_dataset, alpaca_dataset)[:2]
    backend = build_backend(max_length=160, max_new_tokens=24)

    reference_losses = backend.compute_losses(message_dataset.to_records())
    enriched_records = []
    for record, reference_loss in zip(message_dataset, reference_losses, strict=True):
        enriched_records.append(
            set_field(record, "metadata.runtime.reference_loss", float(reference_loss))
        )

    from dataos_operator import Dataset

    train_dataset = Dataset.from_records(enriched_records, metadata=dict(message_dataset.metadata))
    export_sharegpt_dataset(
        train_dataset,
        bundle_dir,
        dataset_name="dataos_policy_runtime_demo",
        split_name="train",
    )

    eval_path = bundle_dir / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for record in eval_dataset:
            f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")

    dataos_config_path = bundle_dir / "dataos_runtime_config.json"
    dataos_config = {
        "mode": "online",
        "update_interval": 1,
        "train_dataset_paths": [str(bundle_dir / "train.jsonl")],
        "eval_dataset_paths": [str(eval_path)],
        "operators": [
            {
                "operator": "loss_selector",
                "kwargs": {
                    "sample_size": 3,
                    "focus": "high",
                    "random_state": 31,
                },
            },
            {
                "operator": "loss_reweighter",
                "kwargs": {
                    "strategy": "linupper",
                },
            },
            {
                "operator": "doremi_mixer",
                "kwargs": {
                    "domain_key": "dataos_domain",
                    "eta": 0.5,
                    "epsilon": 0.05,
                },
            },
        ],
        "backend": {
            "max_length": 160,
            "max_new_tokens": 24,
            "parameter_filters": [],
        },
    }
    dataos_config_path.write_text(
        json.dumps(dataos_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    model_name_or_path = os.environ.get("DATAOS_OPERATOR_DEMO_MODEL", DEFAULT_MODEL)
    train_args = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": model_name_or_path,
        "dataset_dir": str(bundle_dir),
        "dataset": "dataos_policy_runtime_demo",
        "template": "qwen",
        "finetuning_type": "lora",
        "cutoff_len": 160,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "max_steps": 2,
        "save_steps": 2,
        "logging_steps": 1,
        "overwrite_output_dir": True,
        "output_dir": str(bundle_dir / "lf_output"),
        "plot_loss": False,
        "report_to": [],
        "dataos_enable": True,
        "dataos_config_path": str(dataos_config_path),
        "dataos_update_interval": 1,
    }
    if os.environ.get("DATAOS_OPERATOR_DEMO_DEVICE", "").startswith("cuda"):
        train_args["bf16"] = True

    run_exp(train_args)

    summary = {
        "datasets": {
            "gsm8k_path": str(gsm8k_dataset.metadata["path"]),
            "alpaca_path": str(alpaca_dataset.metadata["path"]),
        },
        "bundle_dir": str(bundle_dir),
        "dataos_config_path": str(dataos_config_path),
        "lf_output_dir": str(bundle_dir / "lf_output"),
        "dataos_summary_path": str(bundle_dir / "lf_output" / "dataos" / "summary.json"),
    }
    write_summary(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
