#!/usr/bin/env python3
"""Deterministic source profiler for the RECON stage.

Given a fixed-pool source (or --all-sources), sample rows and produce a structured
``source_profile`` the main-agent uses to pick a data source WITHOUT guessing:

    {source, n, schema[], domain, sample,
     output_format: {function_call_json, natural_language, code, other},   # fractions
     target_fit: {target_ratio, negatives, format_matches_eval}}           # vs the run target

This is PURE STATISTICS — no LLM, no judgment, and deliberately NO quality verdict or
usage hint: the leader judges sources from the raw numbers + samples, a script's label
would bias its planning. Cheap and reliable enough to profile the whole pool in a few
minutes. It exists because the operator box has no "is this a valid target-format
sample?" filter, so a random_sample recipe silently trained on 68% non-tool-call
garbage from glaive and scored 4 on BFCL while the clean xlam source reached 83. This
profiler surfaces exactly that difference — as numbers, not verdicts.

Run inside the eval/train conda env (needs `datasets`):
    python profile_source.py --pool-root <pool> --source xlam_function_calling_60k --target function-calling
    python profile_source.py --pool-root <pool> --all-sources --target function-calling
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_MEDICAL_KW = ("patient", "diagnos", "symptom", "disease", "medical", "clinical", "dosage",
               "mg ", "treatment", "medication", "therapy")
_CODE_MARKERS = ("```", "def ", "import ", "class ", "function ", "#include", "public static")


def _load_split(pool_root: str, source: str):
    from datasets import load_from_disk

    ds = load_from_disk(str(Path(pool_root) / source))
    # DatasetDict -> pick 'train' or the first split; Dataset -> use as-is
    if hasattr(ds, "keys") and not hasattr(ds, "features"):
        split = "train" if "train" in ds else next(iter(ds.keys()))
        ds = ds[split]
    return ds


def _extract_output(row: dict) -> str:
    """Get the assistant/target text across alpaca / messages / sharegpt / qa schemas."""
    for k in ("output", "response", "answer", "target", "completion"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for k in ("conversations", "messages"):
        turns = row.get(k)
        if isinstance(turns, list):
            for t in reversed(turns):
                if not isinstance(t, dict):
                    continue
                role = t.get("from") or t.get("role") or ""
                if str(role).lower() in ("gpt", "assistant", "model", "bot"):
                    return str(t.get("value") or t.get("content") or "")
    return ""


def _extract_context(row: dict) -> str:
    """Prompt-side text (instruction/input/system/user) used to detect negatives + tools."""
    parts = []
    for k in ("instruction", "input", "system", "prompt", "query", "question"):
        v = row.get(k)
        if isinstance(v, str):
            parts.append(v)
    for k in ("conversations", "messages"):
        turns = row.get(k)
        if isinstance(turns, list):
            for t in turns:
                if isinstance(t, dict):
                    role = str(t.get("from") or t.get("role") or "").lower()
                    if role in ("system", "human", "user"):
                        parts.append(str(t.get("value") or t.get("content") or ""))
    return "\n".join(parts)


def _is_valid_function_call(out: str) -> bool:
    """output parses as [{"name":.., "arguments":{...}}] (BFCL-style)."""
    s = out.strip()
    # tolerate a <tool_call> wrapper / leading label
    m = re.search(r"\[\s*\{.*\}\s*\]", s, flags=re.DOTALL)
    if not m:
        return False
    try:
        calls = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(calls, list) or not calls:
        return False
    for c in calls:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            return False
        if not isinstance(c.get("arguments", {}), dict):
            return False
    return True


def _looks_like_functioncall_other(out: str) -> bool:
    """A function call in a NON-BFCL shape (e.g. glaive <functioncall> {...args:'string'})."""
    return "<functioncall>" in out or bool(re.search(r'"name"\s*:', out) and '"arguments"' in out)


def _classify_output(out: str) -> str:
    if _is_valid_function_call(out) or _looks_like_functioncall_other(out):
        return "function_call_json"
    if any(m in out for m in _CODE_MARKERS):
        return "code"
    if out.strip():
        return "natural_language"
    return "other"


def _is_negative(ctx: str, out: str) -> bool:
    low = (ctx + "\n" + out).lower()
    return ("no access to external functions" in low
            or "no access to functions" in low
            or "no external functions" in low)


def _domain(fc_ratio: float, sample_out: str, sample_ctx: str) -> str:
    if fc_ratio > 0.25 or "available tools" in sample_ctx.lower() or "<tool" in sample_out.lower():
        return "function-calling"
    if "#### " in sample_out or re.search(r"\\boxed\{", sample_out):
        return "math"
    if any(kw in (sample_ctx + sample_out).lower() for kw in _MEDICAL_KW):
        return "medical"
    if any(m in sample_out for m in _CODE_MARKERS):
        return "code"
    return "general"


def profile_source(pool_root: str, source: str, target: str = "function-calling",
                   n_sample: int = 200, seed: int = 42) -> dict[str, Any]:
    try:
        ds = _load_split(pool_root, source)
    except Exception as e:  # noqa: BLE001 — a source we can't load is IRRELEVANT, not fatal
        return {"source": source, "n": 0, "quality": "IRRELEVANT",
                "usage_hint": f"could not load: {type(e).__name__}: {e}"}
    n = len(ds)
    take = min(n_sample, n)
    try:
        sample = ds.shuffle(seed=seed).select(range(take))
    except Exception:  # noqa: BLE001
        sample = ds.select(range(take))
    rows = [dict(r) for r in sample]
    schema = sorted(rows[0].keys()) if rows else []

    fmt = {"function_call_json": 0, "natural_language": 0, "code": 0, "other": 0}
    valid_fc = neg = 0
    first_out = first_ctx = ""
    for r in rows:
        out = _extract_output(r)
        ctx = _extract_context(r)
        if not first_out:
            first_out, first_ctx = out, ctx
        fmt[_classify_output(out)] += 1
        if _is_valid_function_call(out):
            valid_fc += 1
        if _is_negative(ctx, out):
            neg += 1
    denom = max(len(rows), 1)
    fmt_frac = {k: round(v / denom, 3) for k, v in fmt.items()}
    valid_ratio = round(valid_fc / denom, 3)
    neg_ratio = round(neg / denom, 3)
    dom = _domain(fmt_frac["function_call_json"], first_out, first_ctx)

    # target_fit: how well this source matches THIS run's target task/format.
    if target == "function-calling":
        target_ratio = valid_ratio
        format_matches_eval = valid_ratio > 0.8
    elif target == "math":
        target_ratio = round(sum(1 for r in rows if "#### " in _extract_output(r)) / denom, 3)
        format_matches_eval = target_ratio > 0.5
    else:
        target_ratio = round(1.0 - fmt_frac["other"], 3)
        format_matches_eval = None

    # Raw statistics ONLY — no quality verdict, no usage hint. Judging which source is
    # worth training on is the leader's job (it sees these numbers + the samples);
    # a script's label would bias that judgment.
    return {
        "source": source,
        "n": n,
        "schema": schema,
        "domain": dom,
        "sample": {"context": first_ctx[:200], "output": first_out[:200]},
        "output_format": fmt_frac,
        "target_fit": {"target_ratio": target_ratio, "negatives": neg_ratio,
                       "format_matches_eval": format_matches_eval},
    }


def _iter_sources(pool_root: str) -> list[str]:
    root = Path(pool_root)
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and not p.name.startswith((".", "_")):
            out.append(p.name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--all-sources", action="store_true")
    ap.add_argument("--target", default="function-calling",
                    help="run target task kind for target_fit (function-calling|math|general)")
    ap.add_argument("--n-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.all_sources:
        profiles = [profile_source(args.pool_root, s, args.target, args.n_sample, args.seed)
                    for s in _iter_sources(args.pool_root)]
        json.dump(profiles, sys.stdout, ensure_ascii=False, indent=2)
    elif args.source:
        json.dump(profile_source(args.pool_root, args.source, args.target, args.n_sample, args.seed),
                  sys.stdout, ensure_ascii=False, indent=2)
    else:
        ap.error("give --source <id> or --all-sources")
    print()


if __name__ == "__main__":
    main()
