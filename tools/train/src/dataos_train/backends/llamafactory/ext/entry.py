"""Patched LLaMA-Factory launch entrypoint that registers online operators.

Invoked by the LF backend instead of ``llamafactory-cli train`` when ``online_ops`` is set:

    python -m dataos_train.backends.llamafactory.ext.entry \
        --online-ops '<json>' -- <train_config.json>

It resolves the ops, monkeypatches the trainer class the SFT workflow instantiates with a
subclass wired to the hooks, then delegates to LF's ``run_exp``. torch/llamafactory are
imported lazily here (cluster-only).
"""

from __future__ import annotations

import argparse
import json
import sys

from ....online import resolve_online_ops


def _assert_patchable_trainer(wf) -> None:  # noqa: ANN001
    """Startup assertion for the monkeypatch discipline (挂法 A): the symbol we rebind and
    the ``compute_loss`` signature we override must match the pinned LF version. A clear
    failure here beats a silent no-op patch after an LF upgrade."""
    import inspect

    if not hasattr(wf, "CustomSeq2SeqTrainer"):
        raise RuntimeError(
            "llamafactory.train.sft.workflow has no 'CustomSeq2SeqTrainer' symbol — the LF "
            "version drifted from the pinned one; re-check the symbol path before bumping LF."
        )
    params = inspect.signature(wf.CustomSeq2SeqTrainer.compute_loss).parameters
    if "inputs" not in params:
        raise RuntimeError(
            "CustomSeq2SeqTrainer.compute_loss no longer takes an 'inputs' parameter — the "
            "LF version drifted from the pinned one; re-check the override signature "
            "(ext/trainer.py) before bumping LF."
        )


def _patch_workflow_trainer(online_trainer_cls: type) -> None:
    """Point LF's SFT workflow at our trainer subclass (version-sensitive — one place)."""
    import llamafactory.train.sft.workflow as wf

    # run_sft references the module-level name, so rebinding it here is sufficient.
    wf.CustomSeq2SeqTrainer = online_trainer_cls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataos_train...ext.entry")
    ap.add_argument("--online-ops", dest="online_ops", default="[]")
    ap.add_argument("config", help="path to the LF train config (json/yaml)")
    args = ap.parse_args(argv)

    specs = json.loads(args.online_ops)
    hooks = resolve_online_ops(specs)
    for hook in hooks:
        hook.setup(None)

    from .trainer import make_online_trainer, set_active_hooks

    set_active_hooks(hooks)

    import llamafactory.train.sft.workflow as wf
    from llamafactory.train.tuner import run_exp

    _assert_patchable_trainer(wf)
    base_cls = wf.CustomSeq2SeqTrainer
    _patch_workflow_trainer(make_online_trainer(base_cls))

    # LF reads its config from argv (single yaml/json file).
    sys.argv = ["llamafactory-cli", args.config]
    run_exp()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
