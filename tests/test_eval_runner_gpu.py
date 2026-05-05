"""GPU smoke: ``run_eval_stage`` end-to-end with TransformersAdapter on Qwen2.5-0.5B.

Skipped unless ``DATAOS_EVAL_GPU_MODEL_PATH`` is set and a CUDA device is
available. Set ``HF_ENDPOINT=https://hf-mirror.com`` so lm-eval can fetch
gsm8k from the mirror.
"""

from __future__ import annotations

import os
import time

import pytest

from dataos.core.artifact import EvalArtifact, ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.eval import run_eval_stage

pytest.importorskip("lm_eval")
pytest.importorskip("torch")

import torch  # noqa: E402

MODEL_PATH = os.environ.get("DATAOS_EVAL_GPU_MODEL_PATH")
GSM8K_LIMIT = int(os.environ.get("DATAOS_EVAL_GPU_GSM8K_LIMIT", "10"))
BATCH_SIZE = int(os.environ.get("DATAOS_EVAL_GPU_BATCH", "16"))
MAX_NEW_TOKENS = int(os.environ.get("DATAOS_EVAL_GPU_MAX_NEW_TOKENS", "256"))


@pytest.mark.gpu
@pytest.mark.skipif(not MODEL_PATH, reason="DATAOS_EVAL_GPU_MODEL_PATH not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_run_eval_stage_gsm8k_qwen_05b(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    model_dir = sandbox.artifact_dir("model", "qwen-25-05b")
    model_dir.mkdir(parents=True, exist_ok=True)
    model = ModelArtifact(
        artifact_id="qwen-25-05b",
        name="qwen2.5-0.5b-instruct",
        path=model_dir,
        producer="test",
        meta={"base_model": MODEL_PATH},
    )
    task = TaskTemplate.from_dict(
        {
            "task_id": "task-gpu-runner",
            "name": "gpu-runner",
            "goal": "qwen 0.5b on gsm8k via lm-eval",
            "input_description": "model artifact",
            "output_description": "eval artifact",
            "constraints": {"dataset_format": "hf_datasets"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["eval_model"],
            "train_config": {"mode": "stub"},
            "eval_config": {
                "tasks": ["gsm8k"],
                "limit": GSM8K_LIMIT,
                "num_fewshot": 0,
                "batch_size": BATCH_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "apply_chat_template": True,
                "adapter": "transformers",
            },
        }
    )

    start = time.time()
    artifacts = run_eval_stage([model], sandbox=sandbox, task=task)
    elapsed = time.time() - start

    eval_artifacts = [artifact for artifact in artifacts if isinstance(artifact, EvalArtifact)]
    assert len(eval_artifacts) == 1
    eval_artifact = eval_artifacts[0]

    metrics = eval_artifact.meta["results"]["gsm8k"]
    flexible = metrics.get("exact_match,flexible-extract")
    strict = metrics.get("exact_match,strict-match")
    assert isinstance(flexible, float)
    assert (eval_artifact.path / "results.json").exists()
    assert (eval_artifact.path / "lm_eval_results.json").exists()
    assert (eval_artifact.path / "samples-gsm8k.jsonl").exists()
    print(
        f"\n[gsm8k limit={GSM8K_LIMIT} batch={BATCH_SIZE}] "
        f"flexible={flexible:.3f} strict={strict:.3f} elapsed={elapsed:.1f}s"
    )
