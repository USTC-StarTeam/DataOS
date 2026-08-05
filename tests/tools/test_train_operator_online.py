"""dataos-train ↔ dataos-operator online-op integration: the operator package exposes
its online-capable operators through the ``dataos_train.online_ops`` entry-point group,
and dataos-train resolves them by name from a TrainSpec — no import coupling, no
vendored trainer fork (the old LLaMA-Factory fork was deleted once this path existed)."""

import pytest
from dataos_train.online import probe_online_op, resolve_online_ops


def test_operator_online_op_is_discoverable():
    assert probe_online_op("loss_reweighter") is True  # entry-point metadata scan


def test_resolve_and_call_operator_reweight_hook():
    hooks = resolve_online_ops(
        [{"op": "loss_reweighter", "params": {"strategy": "linupper", "delta": 1.0}}]
    )
    assert len(hooks) == 1
    hook = hooks[0]
    assert hook.hook_point == "reweight"

    losses = [0.5, 1.0, 2.0, 4.0]
    factors = hook({"losses": losses})
    assert len(factors) == len(losses)
    assert all(f > 0 for f in factors)
    assert sum(factors) / len(factors) == pytest.approx(1.0, abs=1e-6)  # train contract

    # linupper up-weights high-loss (hard) samples relative to easy ones
    assert factors[-1] > factors[0]


def test_operator_hook_rejects_wrong_hook_point():
    with pytest.raises(ValueError, match="reweight"):
        resolve_online_ops([{"op": "loss_reweighter", "hook": "select"}])


def test_unknown_online_op_still_raises_keyerror():
    with pytest.raises(KeyError, match="unknown online op"):
        resolve_online_ops([{"op": "nonexistent_op"}])
