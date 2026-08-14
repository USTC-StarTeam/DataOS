"""Online operators — data interacting with the live training loop (hook protocol v1).

This package is **backend-neutral** and **torch-free**: it owns the hook protocol, the
in-module registry, and entry-point discovery. Backends realize hook points in their own
extension layer (e.g. ``backends/llamafactory/ext/``); payloads carry plain Python types
only. The full protocol contract lives in ``docs/online_ops.md``.

Hook protocol v1 (owned here, stable):
  - Hook points: ``reweight`` / ``select`` / ``step``.
  - Ops are built by a factory: ``factory(hook: str | None = None, **params)``. ``hook=None``
    selects the op's own default hook point; an explicitly passed hook point the op does not
    support must raise ``ValueError`` at instantiation.
  - Instances are duck-typed: ``name: str``, ``hook_point: str``, ``setup(ctx)``.
  - ``reweight``: ``__call__(batch: dict) -> list[float]`` — per-sample weight factors with
    mean ~1. Batch keys: ``"losses": list[float]``, ``"row_indices": list[int] | None``,
    ``"domains": None`` (reserved), ``"step": int``, ``"epoch": int``. Fires every batch.
  - ``step``: ``__call__(info: dict) -> None`` with ``info = {"step", "metrics"}`` — observe
    only, each optimizer step.
  - ``select``: ``observe(batch) -> None`` plus
    ``decide(ctx) -> {"indices": list[int] | None, "domain_weights": dict | None}``.
    Called at epoch boundaries: the decision is made at an epoch boundary and takes effect
    from the next epoch; any ``interval`` parameter is counted in epochs. ``ctx`` carries
    ``{"epoch": int, "step": int, "n_rows": int | None}``. Wired in the
    LLaMA-Factory backend via :class:`~dataos_train.online.select.MutableSubsetSampler`
    (the train sampler is wrapped; ``decide`` swaps the allowed-index set) plus an
    ``on_epoch_end`` callback. ``observe(batch)`` receives the same per-sample-loss payload
    as ``reweight`` hooks, every batch.
  - Zero-marginal-cost discipline (a reweight op should cost ~nothing per step) is a
    *policy* concern of the ops themselves — the protocol does not enforce it.

Discovery: in-module registration via :func:`register_online_op`, plus the entry-point
group ``dataos_train.online_ops`` reserved for external packages (e.g. dataos_operator) —
duck-typed, no import coupling.
"""

from __future__ import annotations

from .reference import EpochSubsetSelectHook, LossReweightHook, apply_reweight, reweight_factors
from .registry import (
    known_online_op_names,
    online_op_names,
    probe_online_op,
    register_online_op,
    resolve_online_ops,
)
from .select import MutableSubsetSampler

__all__ = [
    "EpochSubsetSelectHook",
    "LossReweightHook",
    "MutableSubsetSampler",
    "apply_reweight",
    "known_online_op_names",
    "online_op_names",
    "probe_online_op",
    "register_online_op",
    "resolve_online_ops",
    "reweight_factors",
]
