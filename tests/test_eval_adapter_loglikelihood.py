"""Tests for the EvalLoglikelihoodAdapter Protocol and TransformersAdapter
implementation of ``score_continuations``.

The Protocol shape and runtime check are CPU-only. The numeric correctness
check loads Qwen2.5-0.5B-Instruct on a CUDA device; gated by
``DATAOS_EVAL_GPU_MODEL_PATH`` and ``@pytest.mark.gpu``.
"""
from __future__ import annotations

import os

import pytest

from dataos.core.artifact import ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.eval.adapters.base import EvalLoglikelihoodAdapter
from dataos.eval.adapters.transformers import TransformersAdapter
from dataos.eval.types import EvalConfig, EvalRequest, ModelResponse


class _FakeScoringAdapter:
    name = "fake_scorer"

    def __init__(self, payload: list[tuple[float, bool]]):
        self.payload = payload
        self.calls: list[list[tuple[str, str]]] = []

    def score_continuations(self, model_artifact, pairs, config):
        self.calls.append(list(pairs))
        return list(self.payload[: len(pairs)])


class _GenerateOnlyAdapter:
    name = "generate_only"

    def generate_until(self, model_artifact, requests, config):
        return [
            ModelResponse(
                sample_id=request.sample_id,
                index=request.index,
                request_type="generate_until",
                text="",
            )
            for request in requests
        ]


class TestProtocolShape:
    def test_runtime_isinstance_recognizes_score_continuations(self):
        adapter = _FakeScoringAdapter(payload=[(-1.0, True)])
        assert isinstance(adapter, EvalLoglikelihoodAdapter)

    def test_runtime_isinstance_rejects_generate_only_adapter(self):
        adapter = _GenerateOnlyAdapter()
        assert not isinstance(adapter, EvalLoglikelihoodAdapter)

    def test_real_transformers_adapter_satisfies_protocol(self):
        # Does not load any model — just checks method presence.
        assert isinstance(TransformersAdapter(), EvalLoglikelihoodAdapter)


class TestFakeAdapterReturnsCannedScores:
    def test_pair_count_and_payload_propagate(self):
        canned: list[tuple[float, bool]] = [(-1.5, False), (-0.3, True), (0.0, True)]
        adapter = _FakeScoringAdapter(payload=canned)
        scores = adapter.score_continuations(
            model_artifact=None,
            pairs=[("ctx1", "cont1"), ("ctx2", "cont2"), ("ctx3", "cont3")],
            config=EvalConfig(provider="lm_eval"),
        )
        assert scores == canned
        assert len(adapter.calls) == 1
        assert len(adapter.calls[0]) == 3


# ---------------------------------------------------------------------------
# Real model numeric check (skipped without model path / CUDA).
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("DATAOS_EVAL_GPU_MODEL_PATH")

torch_spec = pytest.importorskip("torch")
transformers_spec = pytest.importorskip("transformers")


@pytest.mark.gpu
@pytest.mark.skipif(not MODEL_PATH, reason="DATAOS_EVAL_GPU_MODEL_PATH not set")
@pytest.mark.skipif(not torch_spec.cuda.is_available(), reason="no CUDA device")
def test_transformers_score_continuations_qwen_05b(tmp_path):
    """Qwen2.5-0.5B should assign higher logprob to a coherent continuation
    than to an obviously wrong one for the same context."""
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    model_dir = sandbox.artifact_dir("model", "qwen-05b-loglik")
    model_dir.mkdir(parents=True, exist_ok=True)
    model = ModelArtifact(
        artifact_id="qwen-05b-loglik",
        name="qwen2.5-0.5b-instruct",
        path=model_dir,
        producer="test",
        meta={"base_model": MODEL_PATH},
    )
    adapter = TransformersAdapter()

    context = "The capital of France is"
    pairs = [
        (context, " Paris."),     # correct
        (context, " banana."),    # nonsense
    ]
    config = EvalConfig(
        provider="lm_eval",
        adapter=None,
        request=None,
    )
    scores = adapter.score_continuations(model, pairs, config)

    assert len(scores) == 2
    coherent_logprob, coherent_greedy = scores[0]
    junk_logprob, junk_greedy = scores[1]
    assert isinstance(coherent_logprob, float)
    assert isinstance(coherent_greedy, bool)
    # The coherent continuation must be more probable than the junk one.
    assert coherent_logprob > junk_logprob, (coherent_logprob, junk_logprob)
    # Logprobs must be non-positive.
    assert coherent_logprob <= 0.0
    assert junk_logprob <= 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not MODEL_PATH, reason="DATAOS_EVAL_GPU_MODEL_PATH not set")
@pytest.mark.skipif(not torch_spec.cuda.is_available(), reason="no CUDA device")
def test_transformers_score_continuations_handles_empty_continuation(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    model_dir = sandbox.artifact_dir("model", "qwen-05b-empty")
    model_dir.mkdir(parents=True, exist_ok=True)
    model = ModelArtifact(
        artifact_id="qwen-05b-empty",
        name="qwen2.5-0.5b-instruct",
        path=model_dir,
        producer="test",
        meta={"base_model": MODEL_PATH},
    )
    adapter = TransformersAdapter()
    config = EvalConfig(provider="lm_eval", adapter=None, request=None)
    scores = adapter.score_continuations(model, [("hello", "")], config)
    assert scores == [(0.0, True)]
