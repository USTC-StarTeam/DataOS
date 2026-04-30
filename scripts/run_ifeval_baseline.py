"""Run IFEval against the unmodified base model (no SFT) to capture a baseline.

Builds a synthetic ``ModelArtifact`` pointing at the local Qwen snapshot and
invokes ``run_eval_stage`` directly. Skips planning, cleaning, and training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataos._env import load_dotenv
from dataos.core.artifact import DatasetArtifact, ModelArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage


def _build_sandbox(workspace: Path) -> Sandbox:
    sandbox = Sandbox(workspace / "sandbox")
    sandbox.prepare()
    return sandbox


def _build_model_artifact(sandbox: Sandbox, model_path: Path) -> ModelArtifact:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    artifact_id = make_artifact_id("model")
    target = sandbox.artifact_dir("model", artifact_id)
    return ModelArtifact(
        artifact_id=artifact_id,
        name="qwen2p5-0p5b-instruct-base",
        path=target,
        producer="run_ifeval_baseline",
        meta={
            "mode": "base_model_eval",
            "base_model": str(model_path),
            "source_dataset_id": None,
        },
    )


def _build_dataset_artifact(dataset_path: Path) -> DatasetArtifact:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    artifact = DatasetArtifact(
        artifact_id=make_artifact_id("dataset"),
        name=f"ifeval-{dataset_path.name}",
        path=dataset_path,
        producer="run_ifeval_baseline",
    )
    artifact.refresh_metadata_from_disk()
    return artifact


def _build_task_template(max_samples: int | None, batch_size: int, max_new_tokens: int) -> TaskTemplate:
    eval_config = {
        "provider": "native",
        "task": {"name": "ifeval"},
        "data": {"split": "train", "max_samples": max_samples} if max_samples else {"split": "train"},
        "adapter": {"name": "transformers"},
        "request": {
            "type": "generate_until",
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
        },
    }
    return TaskTemplate.from_dict(
        {
            "task_id": "ifeval-baseline",
            "name": "IFEval baseline",
            "goal": "Measure IFEval prompt-strict accuracy on the base model (no SFT).",
            "input_description": "Hugging Face save_to_disk directory of IFEval prompts.",
            "output_description": "EvalArtifact with prompt-strict and instruction-strict accuracy.",
            "constraints": {"dataset_format": "hf_datasets"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["eval_model"],
            "train_config": {"mode": "stub"},
            "eval_config": eval_config,
        }
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="IFEval baseline runner (no training)")
    parser.add_argument("--workspace", type=Path, required=True, help="Workspace root")
    parser.add_argument("--dataset", type=Path, required=True, help="HF save_to_disk dir of IFEval")
    parser.add_argument("--model", type=Path, required=True, help="Local base model path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    sandbox = _build_sandbox(args.workspace)
    model_artifact = _build_model_artifact(sandbox, args.model)
    model_artifact.save_meta()
    dataset_artifact = _build_dataset_artifact(args.dataset)
    task = _build_task_template(args.max_samples, args.batch_size, args.max_new_tokens)

    output_artifacts = run_eval_stage(
        [dataset_artifact, model_artifact],
        sandbox=sandbox,
        task=task,
    )
    summary_paths = []
    for artifact in output_artifacts:
        if artifact.type == "eval":
            summary = artifact.meta
            summary_paths.append(str(artifact.path))
            print(json.dumps({"score": summary.get("score"), "summary": summary}, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"\neval artifacts written to: {summary_paths}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
