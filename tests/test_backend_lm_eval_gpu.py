"""GPU smoke test: real Qwen2.5-0.5B on lm-eval gsm8k via DataOSLM bridge.

Skipped unless ``DATAOS_EVAL_GPU_MODEL_PATH`` is set to a local HF snapshot dir.
Set ``HF_ENDPOINT=https://hf-mirror.com`` so lm-eval can fetch the gsm8k dataset.
"""
from __future__ import annotations

import os
import time

import pytest

from dataos.core.artifact import ModelArtifact
from dataos.core.sandbox import Sandbox

pytest.importorskip("lm_eval")
pytest.importorskip("torch")

import torch  # noqa: E402

from dataos.eval.adapters.transformers import TransformersAdapter  # noqa: E402
from dataos.eval.backend import run_lm_eval  # noqa: E402


MODEL_PATH = os.environ.get("DATAOS_EVAL_GPU_MODEL_PATH")
GSM8K_LIMIT = int(os.environ.get("DATAOS_EVAL_GPU_GSM8K_LIMIT", "50"))
BATCH_SIZE = int(os.environ.get("DATAOS_EVAL_GPU_BATCH", "32"))
MAX_NEW_TOKENS = int(os.environ.get("DATAOS_EVAL_GPU_MAX_NEW_TOKENS", "256"))


@pytest.mark.gpu
@pytest.mark.skipif(not MODEL_PATH, reason="DATAOS_EVAL_GPU_MODEL_PATH not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_qwen25_0_5b_runs_gsm8k_on_gpu(tmp_path):
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

    adapter = TransformersAdapter()
    start = time.time()
    output = run_lm_eval(
        model,
        adapter,
        tasks=["gsm8k"],
        limit=GSM8K_LIMIT,
        num_fewshot=0,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        apply_chat_template=True,
    )
    elapsed = time.time() - start

    assert "gsm8k" in output.results
    metrics = output.results["gsm8k"]
    flexible = metrics.get("exact_match,flexible-extract")
    strict = metrics.get("exact_match,strict-match")
    assert isinstance(flexible, float), f"unexpected metrics: {metrics}"
    samples = output.samples["gsm8k"]
    unique_docs = {sample.get("doc_id") for sample in samples}
    assert len(unique_docs) == GSM8K_LIMIT
    print(
        f"\n[gsm8k limit={GSM8K_LIMIT} batch={BATCH_SIZE}] "
        f"flexible={flexible:.3f} strict={strict:.3f} elapsed={elapsed:.1f}s"
    )
