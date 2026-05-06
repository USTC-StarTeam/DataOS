"""Deprecated as the default documented demo.

README.zh-CN.md now treats scripts/run_ifeval_baseline.py as the default demo
entrypoint on the zxy branch. This script is kept for ad hoc runtime checks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, EvalArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage
from dataos.operators.dedup import DedupOperator
from dataos.operators.sample import SampleOperator
from dataos.train import run_train_stage


def _build_demo_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "instruction": [
                        "What is 1 + 1? Answer with only the final number.",
                        "What is 1 + 1? Answer with only the final number.",
                        "What is 2 + 3? Answer with only the final number.",
                        "What is 7 - 4? Answer with only the final number.",
                        "What is 3 * 2? Answer with only the final number.",
                    ],
                    "response": ["2", "2", "5", "3", "6"],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "instruction": [
                        "What is 4 + 4? Answer with only the final number.",
                        "What is 9 - 5? Answer with only the final number.",
                    ],
                    "response": ["8", "4"],
                }
            ),
        }
    )


def _build_task(base_model: Path) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "real-pipeline-demo",
            "name": "Real Pipeline Demo",
            "goal": "Run a verified non-stub DataOS runtime demo end to end.",
            "input_description": "A tiny arithmetic Hugging Face DatasetDict in save_to_disk format.",
            "output_description": "Deduped data, sampled data, a real LoRA adapter, and a real eval report.",
            "constraints": {
                "dataset_format": "hf_datasets",
                "execution_mode": "fixed_runtime",
                "llm_role": ["plan", "review"],
            },
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["dedup", "sample"],
            "train_config": {
                "mode": "llamafactory",
                "model_name_or_path": str(base_model),
                "template": "qwen",
                "finetuning_type": "lora",
                "extra_args": {
                    "cutoff_len": 128,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1.0e-4,
                    "num_train_epochs": 1.0,
                    "logging_steps": 1,
                    "save_steps": 1,
                },
            },
            "eval_config": {
                "provider": "native",
                "task": {"name": "gsm8k"},
                "data": {"split": "test", "max_samples": 2},
                "adapter": {"name": "transformers"},
                "request": {
                    "type": "generate_until",
                    "batch_size": 1,
                    "max_new_tokens": 16,
                    "do_sample": False,
                },
                "artifacts": {"persist_samples": True},
            },
        }
    )


def run_demo(workspace: Path, base_model: Path, overwrite: bool) -> dict[str, object]:
    if overwrite and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    sandbox = Sandbox(workspace / "sandbox")
    sandbox.prepare()

    dataset = _build_demo_dataset()
    dataset_path = workspace / "input_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset.save_to_disk(str(dataset_path))

    initial = DatasetArtifact(
        artifact_id="dataset-demo-initial",
        name="demo-input",
        path=dataset_path,
        producer="run_real_pipeline_demo",
    )
    initial.refresh_metadata_from_disk()
    initial.save_meta()

    deduped = DedupOperator().run([initial], sandbox=sandbox, keys=["instruction"])
    sampled = SampleOperator().run([deduped], sandbox=sandbox, max_train=4, max_test=2)

    task = _build_task(base_model)
    model = run_train_stage([sampled], sandbox=sandbox, task=task)
    model.save_meta()
    eval_outputs = run_eval_stage([sampled, model], sandbox=sandbox, task=task)
    eval_artifact = next(artifact for artifact in eval_outputs if isinstance(artifact, EvalArtifact))
    sample_artifact = next((artifact for artifact in eval_outputs if artifact.type == "result"), None)

    summary = {
        "workspace": str(workspace),
        "base_model": str(base_model),
        "input_dataset": str(dataset_path),
        "train_rows_before_dedup": 5,
        "train_rows_after_dedup": deduped.meta.get("num_rows", {}).get("train"),
        "train_rows_after_sample": sampled.meta.get("num_rows", {}).get("train"),
        "test_rows_after_sample": sampled.meta.get("num_rows", {}).get("test"),
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
    parser = argparse.ArgumentParser(description="Run a verified non-stub DataOS runtime demo.")
    parser.add_argument("--workspace", type=Path, required=True, help="Workspace root for demo outputs")
    parser.add_argument("--base-model", type=Path, required=True, help="Local Hugging Face model path")
    parser.add_argument("--overwrite", action="store_true", help="Delete the workspace before running")
    args = parser.parse_args()

    base_model = args.base_model.expanduser().resolve()
    if not base_model.exists():
        raise FileNotFoundError(f"Base model path not found: {base_model}")

    summary = run_demo(args.workspace.expanduser().resolve(), base_model, args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
