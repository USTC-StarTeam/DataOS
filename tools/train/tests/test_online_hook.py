"""Online-operator seam: torch-free protocol core + registry + backend routing."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

import dataos_train.online as H
from dataos_train.backends.llamafactory import LlamaFactoryBackend
from dataos_train.spec import SFT, DatasetRef, OnlineOpSpec, TrainSpec


def _batch(losses: list[float]) -> dict:
    return {"losses": losses, "row_indices": None, "domains": None, "step": 0, "epoch": 0}


class _StubStepHook:
    name = "stub_step"
    hook_point = "step"

    def setup(self, ctx) -> None:  # noqa: ANN001
        return None

    def __call__(self, info: dict) -> None:
        return None


def test_reweight_factors_math() -> None:
    assert H.reweight_factors([1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0]
    f = H.reweight_factors([0.1, 0.1, 0.1, 5.0], alpha=1.0)
    assert abs(sum(f) / len(f) - 1.0) < 1e-9  # normalized to mean 1
    assert f[3] > f[0]  # hard sample up-weighted
    assert H.reweight_factors([]) == []
    assert H.reweight_factors([0.0, 0.0]) == [1.0, 1.0]
    assert H.reweight_factors([0.1, 5.0], alpha=0.0) == [1.0, 1.0]


def test_apply_reweight_shifts_signal() -> None:
    losses = [0.1, 0.1, 0.1, 5.0]
    factors = H.apply_reweight(_batch(losses), [H.LossReweightHook(hook="reweight")])
    plain = sum(losses) / len(losses)
    weighted = sum(loss_i * w for loss_i, w in zip(losses, factors, strict=False)) / len(losses)
    assert weighted > plain  # the online operator changes the training signal
    # step hooks contribute no reweighting
    assert H.apply_reweight(_batch(losses), [_StubStepHook()]) == [1.0] * 4


def test_loss_reweight_hook_batch_signature() -> None:
    hook = H.LossReweightHook()
    assert hook.hook_point == "reweight"  # hook=None -> op's own default point
    out = hook(_batch([0.1, 5.0]))
    assert len(out) == 2 and out[1] > out[0]
    with pytest.raises(ValueError, match="reweight"):
        H.LossReweightHook(hook="step")  # explicit unsupported hook fails at instantiation


def test_resolve_online_ops() -> None:
    resolved = H.resolve_online_ops(
        [{"op": "loss_reweight", "hook": "reweight", "params": {"alpha": 2.0}}]
    )
    assert len(resolved) == 1 and resolved[0].alpha == 2.0
    # hook=None (omitted) -> the op's default hook point, no implicit "step" default
    defaulted = H.resolve_online_ops([{"op": "loss_reweight"}])
    assert defaulted[0].hook_point == "reweight"
    with pytest.raises(KeyError):
        H.resolve_online_ops([{"op": "does-not-exist"}])
    # an explicit hook the op does not support surfaces from resolve as ValueError
    with pytest.raises(ValueError):
        H.resolve_online_ops([{"op": "loss_reweight", "hook": "select"}])


def _spec(**kw) -> TrainSpec:  # noqa: ANN001
    base = {
        "name": "exp",
        "base_model": "/m",
        "run_dir": "/r",
        "dataset": DatasetRef(path="/d"),
        "method": SFT(finetuning_type="full", template="qwen3"),
    }
    return TrainSpec(**(base | kw))


def test_validate_rejects_collate_and_unknown_ops() -> None:
    # 'collate' is no longer a hook point — rejected at the spec schema itself
    with pytest.raises(ValidationError):
        OnlineOpSpec(op="loss_reweight", hook="collate")
    # belt-and-braces: a spec bypassing schema validation is still rejected by validate()
    bypassed = OnlineOpSpec.model_construct(op="loss_reweight", hook="collate")
    assert any(
        "unsupported online hook" in e
        for e in LlamaFactoryBackend().validate(_spec(online_ops=[bypassed]))
    )
    # unknown op names fail validate (best-effort resolvability check)
    errors = LlamaFactoryBackend().validate(
        _spec(online_ops=[OnlineOpSpec(op="does-not-exist")])
    )
    assert any("unknown online op" in e for e in errors)
    # known op, hook defaulted -> clean
    assert (
        LlamaFactoryBackend().validate(
            _spec(online_ops=[OnlineOpSpec(op="loss_reweight", hook="reweight")])
        )
        == []
    )


def test_backend_routes_online_to_ext_entry() -> None:
    spec = _spec(
        online_ops=[OnlineOpSpec(op="loss_reweight", hook="reweight", params={"alpha": 1.5})]
    )
    s = LlamaFactoryBackend().build(spec).assets["run_train.sh"]
    assert "dataos_train.backends.llamafactory.ext.entry" in s
    assert "--online-ops" in s and "loss_reweight" in s
    assert "llamafactory-cli train " not in s

    plain = (
        LlamaFactoryBackend()
        .build(
            TrainSpec(
                name="x", base_model="/m", run_dir="/r", dataset=DatasetRef(path="/d"), method=SFT(template="qwen3")
            )
        )
        .assets["run_train.sh"]
    )
    assert "llamafactory-cli train" in plain and "ext.entry" not in plain


def test_mutable_subset_sampler() -> None:
    class _Base:
        def __iter__(self):  # noqa: ANN202
            return iter([3, 1, 4, 1, 5])

    s = H.MutableSubsetSampler(_Base(), n=5)
    assert list(s) == [3, 1, 4, 1, 5] and len(s) == 5  # pass-through by default
    s.set_allowed([1, 5])
    assert list(s) == [1, 1, 5] and len(s) == 2  # filtered, base order preserved
    assert s.allowed == frozenset({1, 5})
    s.set_allowed(None)
    assert list(s) == [3, 1, 4, 1, 5] and len(s) == 5  # restored


def test_epoch_subset_select_hook() -> None:
    hook = H.EpochSubsetSelectHook(fraction=0.5)
    assert hook.hook_point == "select"  # hook=None -> default point
    hook.observe(_batch([1.0, 2.0]))  # no-op, must not raise
    assert hook.decide({"epoch": 1, "step": 10, "n_rows": 100}) == {
        "indices": list(range(50)),
        "domain_weights": None,
    }
    assert hook.decide({"epoch": 1, "step": 10, "n_rows": None}) is None  # size unknown
    assert hook.decide({"epoch": 1, "step": 10, "n_rows": 1})["indices"] == [0]  # min 1 row
    with pytest.raises(ValueError, match="select"):
        H.EpochSubsetSelectHook(hook="reweight")
    with pytest.raises(ValueError, match="fraction"):
        H.EpochSubsetSelectHook(fraction=0.0)
    resolved = H.resolve_online_ops([{"op": "epoch_subset_select", "params": {"fraction": 0.25}}])
    assert resolved[0].fraction == 0.25


def test_package_import_is_torch_free() -> None:
    import dataos_train.backends.llamafactory.ext  # noqa: F401

    assert "torch" not in sys.modules  # heavy deps load only inside the training job
