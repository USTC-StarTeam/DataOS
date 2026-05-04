"""Smoke test: DataOSLM bridge can drive lm-eval's gsm8k task with a fake adapter.

Skips if lm-eval is not installed. The fake adapter returns canned ``#### N``
strings so we exercise the full filter / metric pipeline without loading a model
or needing GPU. The dataset (openai/gsm8k) is fetched from the configured HF
endpoint (set ``HF_ENDPOINT=https://hf-mirror.com`` if behind a firewall).
"""
from __future__ import annotations

import os

import pytest

from dataos.core.artifact import ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.eval.types import EvalConfig, EvalRequest, ModelResponse

pytest.importorskip("lm_eval")

from dataos.eval.backend import LmEvalRunOutput, run_lm_eval  # noqa: E402


class _FakeAdapter:
    name = "fake"

    def __init__(self, answer: str = "42"):
        self.answer = answer
        self.call_count = 0
        self.last_batch_size = 0

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: EvalConfig,
    ) -> list[ModelResponse]:
        self.call_count += 1
        self.last_batch_size = len(requests)
        return [
            ModelResponse(
                sample_id=request.sample_id,
                index=request.index,
                request_type="generate_until",
                text=f"The answer is {self.answer}. #### {self.answer}",
            )
            for request in requests
        ]


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "fake-model-1")
    model_dir.mkdir(parents=True, exist_ok=True)
    return ModelArtifact(
        artifact_id="fake-model-1",
        name="fake",
        path=model_dir,
        producer="test",
        meta={"base_model": "fake/Qwen-stub"},
    )


def test_bridge_runs_gsm8k_with_fake_adapter(tmp_path):
    if not os.environ.get("HF_ENDPOINT") and not os.environ.get(
        "HF_DATASETS_OFFLINE"
    ):
        pytest.skip("HF_ENDPOINT not set; skip to avoid network reach to huggingface.co")
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    model = _model_artifact(sandbox)
    adapter = _FakeAdapter(answer="42")

    output: LmEvalRunOutput = run_lm_eval(
        model,
        adapter,
        tasks=["gsm8k"],
        limit=2,
        num_fewshot=0,
        batch_size=2,
        max_new_tokens=32,
    )

    assert "gsm8k" in output.results, output.results
    gsm8k = output.results["gsm8k"]
    score_keys = [k for k in gsm8k if "exact_match" in k]
    assert score_keys, f"no exact_match metric in {gsm8k}"
    assert adapter.call_count >= 1
    assert "gsm8k" in output.samples
    samples = output.samples["gsm8k"]
    assert samples, "no samples logged"
    unique_doc_ids = {sample.get("doc_id") for sample in samples}
    assert len(unique_doc_ids) == 2, f"expected 2 docs sampled, got {unique_doc_ids}"
    sample = samples[0]
    assert "resps" in sample or "filtered_resps" in sample


def test_bridge_loglikelihood_raises_for_generate_only_adapter():
    """Generate-only adapter should make DataOSLM.loglikelihood raise with a
    clear message; loglikelihood_rolling is unsupported for any adapter."""
    from dataos.eval.backend import DataOSLM

    sandbox_dir = "/tmp/dataos-eval-smoke-not-used"
    model = ModelArtifact(
        artifact_id="x",
        name="x",
        path=sandbox_dir,
        producer="t",
        meta={"base_model": "fake"},
    )
    lm = DataOSLM(model, _FakeAdapter())
    assert lm.supports_loglikelihood() is False
    with pytest.raises(NotImplementedError, match="score_continuations"):
        lm.loglikelihood([])
    with pytest.raises(NotImplementedError):
        lm.loglikelihood_rolling([])


class _ScoringAdapter:
    """Adapter implementing both generate_until and score_continuations."""

    name = "scoring_fake"

    def __init__(self, scores: list[tuple[float, bool]]):
        self.scores = scores
        self.score_calls: list[list[tuple[str, str]]] = []

    def generate_until(self, model, requests, config):
        return [
            ModelResponse(
                sample_id=request.sample_id,
                index=request.index,
                request_type="generate_until",
                text="",
            )
            for request in requests
        ]

    def score_continuations(self, model, pairs, config):
        self.score_calls.append(list(pairs))
        return list(self.scores[: len(pairs)])


def test_bridge_loglikelihood_dispatches_to_scoring_adapter():
    """When the adapter implements score_continuations, DataOSLM.loglikelihood
    must forward (context, continuation) pairs and pass results back as-is."""
    from lm_eval.api.instance import Instance

    from dataos.eval.backend import DataOSLM

    canned: list[tuple[float, bool]] = [(-2.5, True), (-7.0, False)]
    adapter = _ScoringAdapter(scores=canned)
    model = ModelArtifact(
        artifact_id="m",
        name="m",
        path="/tmp/dataos-eval-smoke-not-used",
        producer="t",
        meta={"base_model": "fake"},
    )
    lm = DataOSLM(model, adapter)
    assert lm.supports_loglikelihood() is True

    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("ctx-A", "cont-A"),
            idx=0,
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("ctx-B", "cont-B"),
            idx=1,
        ),
    ]
    out = lm.loglikelihood(requests)

    assert out == canned
    assert adapter.score_calls == [[("ctx-A", "cont-A"), ("ctx-B", "cont-B")]]
