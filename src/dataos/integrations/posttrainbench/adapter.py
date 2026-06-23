"""Fixed PostTrainBench evaluation protocol (agent-version-ptb branch).

The official PTB-fixed-pool task is **per-(base_model, single bench)**: each run optimizes
ONE benchmark for ONE base model (10h / 1 GPU, fixed training pool). So the primary entry is
:func:`ptb_eval` — a single-bench eval config (the official ``src/eval/tasks/<bench>/
evaluate.py`` via the harness, ``--model-path/--limit/--json-output-file``); its score IS the
optimization target / stop criterion. :func:`ptb_suite` (all 7, macro) is kept only for an
optional multi-bench reporting view. The planner shapes training only; never the eval.

A bench whose external key/token is absent at run time fails inside ``run_ptb.sh`` and is
isolated out by DataOS's suite combiner (``skipped_tasks``), so the macro is taken over
whatever actually ran, with a coverage caution. Concretely:
  - no key:        gsm8k, bfcl, humaneval, aime2025            (4)
  - + ~/.mimo_key: healthbench, arenahardwriting (LLM judge)   (-> 6)
  - + HF_TOKEN:    gpqamain (gated Idavidrein/gpqa)            (-> 7)

Running requires the official scorer deployed and pointed at by ``PTB_ROOT`` (vllm 0.11.0,
transformers 4.57.3 — the frozen identity below); ``run_ptb.sh`` self-handles the judge env
(``~/.mimo_key`` + proxy) and LoRA merge cache. This module only *fixes the protocol*.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_HARNESS = _REPO_ROOT / "resources" / "code" / "posttrainbench" / "harness" / "run_ptb.sh"

#: Written into every entry's ``harness`` field — the frozen protocol identity (see README).
PTB_HARNESS_IDENTITY = {
    "repo": "aisa-group/PostTrainBench",
    "commit": "d3496fa7",
    "patch": "judge-model-env",
    "inspect_evals": "bbc1e495",
    "vllm": "0.11.0",
    "transformers": "4.57.3",
    "judge_model": "mimo-v2.5-pro (clash proxy; internal ranking only, not official-board comparable)",
}

PTB_PROFILES = ("formal", "smoke")

# name, bench-arg, (formal limit, formal timeout), (smoke limit, smoke timeout), external need
_BENCHES: list[tuple[str, str, tuple[int, int], tuple[int, int], str | None]] = [
    ("ptb_gsm8k", "gsm8k", (200, 1800), (20, 600), None),
    ("ptb_bfcl", "bfcl", (-1, 10800), (20, 600), None),
    ("ptb_humaneval", "humaneval", (150, 10800), (10, 2400), None),
    ("ptb_aime2025", "aime2025", (-1, 10800), (5, 2400), None),
    ("ptb_healthbench", "healthbench", (32, 10800), (3, 2400), "judge (~/.mimo_key)"),
    ("ptb_arenahardwriting", "arenahardwriting", (32, 10800), (3, 2400), "judge (~/.mimo_key)"),
    ("ptb_gpqamain", "gpqamain", (-1, 10800), (5, 2400), "HF_TOKEN (gated)"),
]

#: Informational map of which benches need an external key/token (run_ptb.sh self-injects it).
PTB_REQUIREMENTS = {name: need for name, _b, _f, _s, need in _BENCHES if need}


PTB_BENCHES = [b for _n, b, _f, _s, _need in _BENCHES]


def _resolve(profile: str, harness: str | None) -> str:
    if profile not in PTB_PROFILES:
        raise ValueError(f"unknown PTB profile {profile!r}; use one of {PTB_PROFILES}")
    return str(harness or os.environ.get("PTB_HARNESS") or _DEFAULT_HARNESS)


def ptb_eval(bench: str, *, profile: str = "formal", harness: str | None = None) -> dict:
    """Single-bench PTB eval config (the per-(base,bench) optimization target).

    Returns a ``provider="external"`` config whose receipt ``score`` is that bench's
    official metric — used directly as the stop criterion."""
    harness_path = _resolve(profile, harness)
    for name, b, formal, smoke, _need in _BENCHES:
        if b == bench:
            limit, timeout = formal if profile == "formal" else smoke
            return {
                "name": name,
                "provider": "external",
                "timeout": timeout,
                "command": f"bash {harness_path} {bench} {{model_path}} {{results_file}} {limit}",
                "harness": dict(PTB_HARNESS_IDENTITY),
            }
    raise ValueError(f"unknown PTB bench {bench!r}; one of {PTB_BENCHES}")


def ptb_suite(*, profile: str = "formal", harness: str | None = None) -> dict:
    """All 7 benches as one suite (macro) — optional multi-bench reporting view."""
    return {"suite": [ptb_eval(b, profile=profile, harness=harness) for b in PTB_BENCHES]}
