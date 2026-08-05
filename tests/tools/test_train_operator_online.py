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


# --- select-hook bridges (loss_selector / delta_loss_selector) ----------------

N_ROWS = 10
LOSSES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # row i has loss (i+1)/10


def _observe_epoch(hook, rows=N_ROWS, losses=LOSSES):
    """Simulate one epoch: two batches covering every row once."""
    hook.observe({"losses": losses[:5], "row_indices": list(range(5))})
    hook.observe({"losses": losses[5:], "row_indices": list(range(5, rows))})


def test_loss_selector_bridge_focuses_high_loss_rows():
    hook = resolve_online_ops(
        [{"op": "loss_selector",
          "params": {"sample_size": 4, "focus": "high", "focus_weight": 50.0,
                     "random_state": 42}}]
    )[0]
    assert hook.hook_point == "select"
    _observe_epoch(hook)
    decision = hook.decide({"epoch": 1, "step": 100, "n_rows": N_ROWS})
    assert decision is not None and decision["domain_weights"] is None
    picked = decision["indices"]
    assert len(picked) == 4 and all(0 <= i < N_ROWS for i in picked)
    # focus=high with a heavy weight must pick overwhelmingly from the top losses
    assert sum(1 for i in picked if i >= 6) >= 3


def test_loss_selector_without_observations_passes_through():
    hook = resolve_online_ops(
        [{"op": "loss_selector", "params": {"sample_size": 4, "random_state": 42}}]
    )[0]
    # nothing observed -> no losses accumulated -> no decision (sampler untouched)
    assert hook.decide({"epoch": 1, "step": 0, "n_rows": N_ROWS}) is None


def test_delta_loss_selector_bridge_flow():
    hook = resolve_online_ops(
        [{"op": "delta_loss_selector",
          "params": {"sample_size": 3, "window_fraction": 0.5, "random_state": 42}}]
    )[0]
    _observe_epoch(hook)  # epoch 1: initial losses
    first = hook.decide({"epoch": 1, "step": 50, "n_rows": N_ROWS})
    _observe_epoch(hook, losses=[x * 0.5 for x in LOSSES])  # epoch 2: everything improved
    second = hook.decide({"epoch": 2, "step": 100, "n_rows": N_ROWS})
    assert first is not None and len(first["indices"]) == 3
    assert second is not None and len(second["indices"]) == 3


def test_select_bridges_reject_wrong_hook_point():
    for op in ("loss_selector", "delta_loss_selector"):
        with pytest.raises(ValueError, match="select"):
            resolve_online_ops([{"op": op, "hook": "reweight",
                                 "params": {"sample_size": 3}}])
