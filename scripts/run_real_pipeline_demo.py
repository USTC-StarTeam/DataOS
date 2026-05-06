"""Run a real AlpacaEval -> operators -> SFT -> IFEval demo.

This script consumes real local source files instead of synthetic toy rows:

- ``alpaca_eval.json`` from ``tatsu-lab/alpaca_eval`` for SFT input.
- a JSON list of real ``google/IFEval`` rows for evaluation.

The runtime path is fully non-stub:

1. Normalize AlpacaEval into a Hugging Face ``DatasetDict``.
2. Import it through DataOS ``import_dataset``.
3. Process it through real ``dedup`` and ``sample`` operators.
4. Run real LoRA SFT through ``LLaMA-Factory``.
5. Evaluate the trained adapter on real IFEval prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import EvalArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.operators.dedup import DedupOperator
from dataos.operators.import_dataset import ImportDatasetOperator
from dataos.operators.sample import SampleOperator
from dataos.train import run_train_stage


def _load_json_list(path: Path, *, label: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{label} row {index} must be a JSON object")
        rows.append(item)
    if not rows:
        raise ValueError(f"{label} must not be empty: {path}")
    return rows


def _normalize_alpaca_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        instruction = row.get("instruction")
        response = row.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"alpaca_eval row {index} is missing a non-empty 'instruction'")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"alpaca_eval row {index} is missing a non-empty 'output'")
        normalized.append(
            {
                "instruction": instruction.strip(),
                "input": "",
                "response": response.strip(),
                "source_dataset": str(row.get("dataset") or ""),
                "generator": str(row.get("generator") or ""),
            }
        )
    return normalized


def _normalize_ifeval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        instruction_ids = row.get("instruction_id_list")
        kwargs_list = row.get("kwargs")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"ifeval row {index} is missing a non-empty 'prompt'")
        if not isinstance(instruction_ids, list) or not instruction_ids:
            raise ValueError(f"ifeval row {index} is missing a non-empty 'instruction_id_list'")
        if not isinstance(kwargs_list, list):
            raise ValueError(f"ifeval row {index} is missing list field 'kwargs'")
        normalized.append(
            {
                "key": str(row.get("key", index)),
                "prompt": prompt,
                "instruction_id_list": instruction_ids,
                "kwargs": kwargs_list,
            }
        )
    return normalized


def _materialize_alpaca_eval_dataset(
    alpaca_eval_json: Path,
    workspace: Path,
    *,
    validation_ratio: float,
    split_seed: int,
) -> tuple[Path, int]:
    rows = _load_json_list(alpaca_eval_json, label="alpaca_eval.json")
    normalized = _normalize_alpaca_eval_rows(rows)
    dataset = Dataset.from_list(normalized)
    split = dataset.train_test_split(test_size=validation_ratio, seed=split_seed)
    dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})
    target = workspace / "source_datasets" / "alpaca_eval_hf"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(target))
    return target, len(normalized)


def _materialize_ifeval_dataset(ifeval_json: Path, workspace: Path) -> tuple[Path, int]:
    rows = _load_json_list(ifeval_json, label="ifeval_rows.json")
    normalized = _normalize_ifeval_rows(rows)
    dataset_dict = DatasetDict({"train": Dataset.from_list(normalized)})
    target = workspace / "source_datasets" / "ifeval_hf"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(target))
    return target, len(normalized)


def _build_task(base_model: Path, ifeval_dataset_path: Path, max_eval_samples: int) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "real-alpaca-ifeval-demo",
            "name": "Real AlpacaEval -> IFEval Demo",
            "goal": "Run a verified non-stub DataOS runtime demo end to end on real source data.",
            "input_description": (
                "Real AlpacaEval rows normalized into Hugging Face train/validation splits, then "
                "processed by DataOS operators before SFT."
            ),
            "output_description": (
                "Imported/processed datasets, a real LoRA adapter, and a real IFEval EvalArtifact."
            ),
            "constraints": {
                "dataset_format": "hf_datasets",
                "execution_mode": "fixed_runtime",
                "llm_role": ["plan", "review"],
            },
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["import_dataset", "dedup", "sample"],
            "train_config": {
                "mode": "llamafactory",
                "model_name_or_path": str(base_model),
                "template": "qwen",
                "finetuning_type": "lora",
                "extra_args": {
                    "bf16": True,
                    "cutoff_len": 512,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1.0e-4,
                    "num_train_epochs": 1.0,
                    "logging_steps": 1,
                    "save_steps": 4,
                    "warmup_ratio": 0.03,
                },
            },
            "eval_config": {
                "provider": "native",
                "task": {"name": "ifeval"},
                "data": {
                    "path": str(ifeval_dataset_path),
                    "split": "train",
                    "max_samples": max_eval_samples,
                },
                "adapter": {"name": "transformers"},
                "request": {
                    "type": "generate_until",
                    "batch_size": 1,
                    "max_new_tokens": 256,
                    "do_sample": False,
                },
                "artifacts": {"persist_samples": True},
            },
        }
    )


def run_demo(
    workspace: Path,
    *,
    base_model: Path,
    alpaca_eval_json: Path,
    ifeval_json: Path,
    max_train_samples: int,
    max_validation_samples: int,
    max_eval_samples: int,
    overwrite: bool,
    validation_ratio: float,
    split_seed: int,
) -> dict[str, object]:
    if overwrite and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    sandbox = Sandbox(workspace / "sandbox")
    sandbox.prepare()

    alpaca_dataset_path, alpaca_row_count = _materialize_alpaca_eval_dataset(
        alpaca_eval_json,
        workspace,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
    )
    ifeval_dataset_path, ifeval_row_count = _materialize_ifeval_dataset(ifeval_json, workspace)

    imported = ImportDatasetOperator().run(
        [],
        sandbox=sandbox,
        source="hf_local",
        path=str(alpaca_dataset_path),
        name="alpaca-eval-normalized",
    )
    deduped = DedupOperator().run([imported], sandbox=sandbox, keys=["instruction"])
    sampled = SampleOperator().run(
        [deduped],
        sandbox=sandbox,
        max_train=max_train_samples,
        max_per_split={"validation": max_validation_samples},
    )

    task = _build_task(base_model, ifeval_dataset_path, max_eval_samples=max_eval_samples)
    model = run_train_stage([sampled], sandbox=sandbox, task=task)
    model.save_meta()
    eval_outputs = run_eval_stage([sampled, model], sandbox=sandbox, task=task)
    eval_artifact = next(artifact for artifact in eval_outputs if isinstance(artifact, EvalArtifact))
    sample_artifact = next((artifact for artifact in eval_outputs if artifact.type == "result"), None)

    summary = {
        "workspace": str(workspace),
        "base_model": str(base_model),
        "alpaca_eval_json": str(alpaca_eval_json),
        "ifeval_json": str(ifeval_json),
        "normalized_alpaca_dataset": str(alpaca_dataset_path),
        "normalized_ifeval_dataset": str(ifeval_dataset_path),
        "alpaca_eval_row_count": alpaca_row_count,
        "ifeval_row_count": ifeval_row_count,
        "imported_num_rows": imported.meta.get("num_rows"),
        "train_rows_after_dedup": deduped.meta.get("num_rows", {}).get("train"),
        "validation_rows_after_dedup": deduped.meta.get("num_rows", {}).get("validation"),
        "train_rows_after_sample": sampled.meta.get("num_rows", {}).get("train"),
        "validation_rows_after_sample": sampled.meta.get("num_rows", {}).get("validation"),
        "dedup_removed_counts": deduped.meta.get("dedup_removed_counts"),
        "sample_kept_counts": sampled.meta.get("sample_kept_counts"),
        "model_artifact": str(model.path),
        "train_log": model.meta.get("log_path"),
        "eval_artifact": str(eval_artifact.path),
        "eval_score": eval_artifact.meta.get("score"),
        "eval_metric": eval_artifact.meta.get("metric"),
        "sample_artifact": str(sample_artifact.path) if sample_artifact is not None else None,
        "eval_summary": eval_artifact.meta,
    }
    summary_path = workspace / "real_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real AlpacaEval -> operators -> LLaMA-Factory -> IFEval demo."
    )
    parser.add_argument("--workspace", type=Path, required=True, help="Workspace root for demo outputs")
    parser.add_argument("--base-model", type=Path, required=True, help="Local Hugging Face model path")
    parser.add_argument(
        "--alpaca-eval-json",
        type=Path,
        required=True,
        help="Local raw alpaca_eval.json downloaded from tatsu-lab/alpaca_eval",
    )
    parser.add_argument(
        "--ifeval-json",
        type=Path,
        required=True,
        help="Local JSON list of real IFEval rows",
    )
    parser.add_argument("--max-train-samples", type=int, default=8)
    parser.add_argument("--max-validation-samples", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Delete the workspace before running")
    args = parser.parse_args()

    base_model = args.base_model.expanduser().resolve()
    if not base_model.exists():
        raise FileNotFoundError(f"Base model path not found: {base_model}")

    alpaca_eval_json = args.alpaca_eval_json.expanduser().resolve()
    if not alpaca_eval_json.exists():
        raise FileNotFoundError(f"AlpacaEval JSON path not found: {alpaca_eval_json}")

    ifeval_json = args.ifeval_json.expanduser().resolve()
    if not ifeval_json.exists():
        raise FileNotFoundError(f"IFEval JSON path not found: {ifeval_json}")

    summary = run_demo(
        args.workspace.expanduser().resolve(),
        base_model=base_model,
        alpaca_eval_json=alpaca_eval_json,
        ifeval_json=ifeval_json,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        max_eval_samples=args.max_eval_samples,
        overwrite=args.overwrite,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
