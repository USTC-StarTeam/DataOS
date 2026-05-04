"""Baseline run: Qwen2.5-0.5B-Instruct on a representative lm-eval task suite.

Manual / non-regression. Runs only when ``DATAOS_EVAL_GPU_MODEL_PATH`` is set.
Each task runs through ``run_eval_stage`` (the runtime path) so we exercise the
full DataOS surface, not just the bridge. Picks generate_until tasks because
``DataOSLM`` does not yet implement loglikelihood (multiple-choice tasks like
mmlu/arc/hellaswag would require it).
"""
from __future__ import annotations

import json
import os
import time

import pytest

from dataos.core.artifact import EvalArtifact, ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage

pytest.importorskip("lm_eval")
pytest.importorskip("torch")
pytest.skip("phase B reverted: relies on new EvalConfig schema (tasks/adapter)", allow_module_level=True)

import torch  # noqa: E402

MODEL_PATH = os.environ.get("DATAOS_EVAL_GPU_MODEL_PATH")
RESULTS_OUT = os.environ.get(
    "DATAOS_EVAL_BASELINE_OUT", "/tmp/dataos-eval-baseline-qwen.json"
)


# (task_name, limit, batch_size, max_new_tokens, num_fewshot, confirm_unsafe)
BASELINE_TASKS: list[tuple[str, int, int, int, int, bool]] = [
    ("gsm8k", 200, 64, 256, 0, False),
    ("ifeval", 100, 32, 1280, 0, False),
    ("humaneval", 80, 32, 768, 0, True),
    ("bbh_cot_zeroshot_logical_deduction_three_objects", 100, 64, 512, 0, False),
]


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "qwen-25-05b")
    model_dir.mkdir(parents=True, exist_ok=True)
    return ModelArtifact(
        artifact_id="qwen-25-05b",
        name="qwen2.5-0.5b-instruct",
        path=model_dir,
        producer="test",
        meta={"base_model": MODEL_PATH},
    )


def _task(
    task_name: str,
    *,
    limit: int,
    batch_size: int,
    max_new_tokens: int,
    num_fewshot: int,
    confirm_unsafe: bool,
) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": f"task-baseline-{task_name}",
            "name": f"baseline-{task_name}",
            "goal": "qwen baseline",
            "input_description": "model artifact",
            "output_description": "eval artifact",
            "constraints": {"dataset_format": "hf_datasets"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["eval_model"],
            "train_config": {"mode": "stub"},
            "eval_config": {
                "tasks": [task_name],
                "adapter": {"name": "transformers"},
                "limit": limit,
                "num_fewshot": num_fewshot,
                "request": {"batch_size": batch_size, "max_new_tokens": max_new_tokens},
                "apply_chat_template": True,
                "confirm_run_unsafe_code": confirm_unsafe,
            },
        }
    )


def _flatten_metrics(results: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for task_name, metrics in (results or {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and "stderr" not in key and key != "alias":
                flat[f"{task_name}::{key}"] = float(value)
    return flat


@pytest.mark.gpu
@pytest.mark.skipif(not MODEL_PATH, reason="DATAOS_EVAL_GPU_MODEL_PATH not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_qwen_baseline_suite(tmp_path):
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows: list[dict] = []
    for task_name, limit, batch_size, max_new_tokens, num_fewshot, confirm_unsafe in BASELINE_TASKS:
        task_root = tmp_path / task_name
        sandbox = Sandbox(task_root / "sandbox")
        sandbox.prepare()
        model = _model_artifact(sandbox)
        task = _task(
            task_name,
            limit=limit,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            num_fewshot=num_fewshot,
            confirm_unsafe=confirm_unsafe,
        )
        start = time.time()
        try:
            artifacts = run_eval_stage([model], sandbox=sandbox, task=task)
            elapsed = time.time() - start
            eval_artifact = next(a for a in artifacts if isinstance(a, EvalArtifact))
            metrics = _flatten_metrics(eval_artifact.meta["results"])
            row = {
                "task": task_name,
                "limit": limit,
                "batch_size": batch_size,
                "elapsed_s": round(elapsed, 1),
                "metrics": metrics,
                "headline_score": eval_artifact.meta.get("score"),
                "headline_metric": eval_artifact.meta.get("metric"),
            }
        except Exception as exc:  # noqa: BLE001 - baseline keeps going
            row = {
                "task": task_name,
                "limit": limit,
                "batch_size": batch_size,
                "elapsed_s": round(time.time() - start, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(f"\n[BASELINE] {task_name}: {row}")

    out = {"model": MODEL_PATH, "results": rows}
    with open(RESULTS_OUT, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False, default=str)
    print(f"\n[BASELINE] wrote {RESULTS_OUT}")
    assert all("error" not in row for row in rows), [r for r in rows if "error" in r]
