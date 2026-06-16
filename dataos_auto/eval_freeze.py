"""Freeze the evaluation protocol ONCE, before the first round.

Every experiment in every round is scored with the same frozen eval config — the loop
overwrites each plan spec's ``eval`` with this value so the planner can never drift the
protocol (which would make rankings meaningless; the library would flag it via
``study.summary()["caution"]``, but we prevent it up front).

MVP: the frozen config is supplied explicitly (``cfg.eval_cfg``) for real runs, or a
stub score for ``cfg.stub``. Autonomous eval *discovery* (an LLM reading a scenario's
official scorers and validating against a known baseline) is deferred — the ``agent``
parameter is reserved for it.
"""

from __future__ import annotations

from .config import AutoRunConfig


def freeze_eval(*, cfg: AutoRunConfig, agent=None) -> dict:
    """Return the single shared eval config used for the whole project.

    On agent-version-ptb the evaluation is FIXED to PostTrainBench — ``cfg.eval_cfg`` is
    ignored. With ``cfg.target_bench`` set (the per-(base,bench) protocol) it returns that
    single bench's official eval; otherwise the 7-bench suite (macro) as a reporting view.
    ``cfg.stub`` still short-circuits to a stub score for tests/dev.
    """
    if cfg.stub:
        return {"provider": "stub", "provider_config": {"score": 0.5}}
    from .ptb import ptb_eval, ptb_suite

    if cfg.target_bench:
        return ptb_eval(cfg.target_bench, profile=cfg.ptb_profile)
    return ptb_suite(profile=cfg.ptb_profile)
