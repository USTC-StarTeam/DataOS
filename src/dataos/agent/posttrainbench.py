"""Fixed PostTrainBench evaluation protocol (agent-version-ptb branch).

The official PTB-fixed-pool task is **per-(base_model, single bench)**: each run optimizes ONE
benchmark for ONE base model (10h / 1 GPU, fixed training pool). :func:`ptb_eval` returns a
single-bench eval config (bench + sampled limit + profile); its score IS the optimization
target / stop criterion. :func:`ptb_suite` (all 7, macro) is an optional multi-bench reporting
view. The planner shapes training only; never the eval.

Scoring runs through the self-contained lm-evaluation-harness bridge, which now lives in the
agent-agnostic runtime repo (``DataOS-Runtime/tasks/posttrainbench/eval``, read from
``runtime_root`` on the exec host) — 5 deterministic benches scored in-job, 2 judge benches
(healthbench, arenahardwriting) generated in-job then judged on the login node. This module only
*fixes the protocol* (which benches, the sampled limit per profile); the sub-agent runs the
bridge per the runtime task contract (``task.yaml`` scoring, rendered into its prompt).
"""

from __future__ import annotations

PTB_PROFILES = ("formal", "smoke")

# name, bench-arg, (formal limit, formal timeout), (smoke limit, smoke timeout), external need
_BENCHES: list[tuple[str, str, tuple[int, int], tuple[int, int], str | None]] = [
    ("ptb_gsm8k", "gsm8k", (200, 1800), (20, 600), None),
    ("ptb_bfcl", "bfcl", (-1, 10800), (20, 600), None),
    ("ptb_humaneval", "humaneval", (150, 10800), (10, 2400), None),
    ("ptb_aime2025", "aime2025", (-1, 10800), (5, 2400), None),
    ("ptb_healthbench", "healthbench", (32, 10800), (3, 2400), "judge (login-node Bailian)"),
    ("ptb_arenahardwriting", "arenahardwriting", (32, 10800), (3, 2400), "judge (login-node Bailian)"),
    ("ptb_gpqamain", "gpqamain", (-1, 10800), (5, 2400), "HF_TOKEN (gated)"),
]

#: Which benches need an external key/token at judge/eval time (informational).
PTB_REQUIREMENTS = {name: need for name, _b, _f, _s, need in _BENCHES if need}

PTB_BENCHES = [b for _n, b, _f, _s, _need in _BENCHES]


def ptb_eval(bench: str, *, profile: str = "formal") -> dict:
    """Single-bench PTB eval config (the per-(base,bench) optimization target).

    The receipt ``score`` is that bench's lm-eval metric (native percentage), used directly as
    the stop criterion. ``provider="lm_eval"`` marks the bridge as the scorer."""
    if profile not in PTB_PROFILES:
        raise ValueError(f"unknown PTB profile {profile!r}; use one of {PTB_PROFILES}")
    for name, b, formal, smoke, _need in _BENCHES:
        if b == bench:
            limit, timeout = formal if profile == "formal" else smoke
            return {
                "name": name,
                "provider": "lm_eval",
                "bench": bench,
                "profile": profile,
                "limit": limit,
                "timeout": timeout,
            }
    raise ValueError(f"unknown PTB bench {bench!r}; one of {PTB_BENCHES}")


def ptb_suite(*, profile: str = "formal") -> dict:
    """All 7 benches as one suite (macro) — optional multi-bench reporting view."""
    return {"suite": [ptb_eval(b, profile=profile) for b in PTB_BENCHES]}
