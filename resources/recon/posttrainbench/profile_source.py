#!/usr/bin/env python3
"""Deterministic source profiler for the RECON stage.

Given a fixed-pool source (or --all-sources), sample rows and produce a structured
``source_profile`` the main-agent uses to pick a data source WITHOUT guessing:

    {source, n_rows, n_sampled, schema[],
     domain: {label, scores: {function-calling, math, medical, code}},  # full-sample evidence
     sample,                                                            # one real row (truncated)
     output_format: {function_call_json, natural_language, code, other}, # fractions
     negatives}                                                          # refusal-rate

This is PURE STATISTICS — no LLM, no judgment: NO quality verdict, NO usage hint, and
NO target_fit/format-fit score (format fit is a fuzzy, bench-specific judgment a regex
can only pretend to make — it once scored the genuinely-math openmathinstruct2 as 0.0).
The INVESTIGATOR writes a per-source textual note over these stats + samples (it holds
the eval format contract); the leader judges from stats + note. Cheap enough to profile
the whole pool in a few minutes. It exists because the operator box has no "is this a
valid target-format sample?" filter, so a random_sample recipe silently trained on 68%
non-tool-call garbage from glaive and scored 4 on BFCL while the clean xlam source
reached 83. This profiler surfaces exactly that difference — as numbers, not verdicts.

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


# Layer 1 key lists — fast path for the well-known SFT schemas. Anything not matched here
# falls through to structural inference (layer 2), never to silent empties.
_OUT_KEYS = ("output", "response", "answer", "target", "completion",
             "generated_solution", "solution", "expected_answer", "chosen", "content", "text")
_CTX_KEYS = ("instruction", "input", "system", "prompt", "query", "question", "problem")


def _from_turns(row: dict, roles: tuple[str, ...]) -> str:
    """sharegpt/messages turn extraction (reversed order for the target side)."""
    for k in ("conversations", "messages"):
        turns = row.get(k)
        if isinstance(turns, list):
            for t in reversed(turns):
                if not isinstance(t, dict):
                    continue
                role = str(t.get("from") or t.get("role") or "").lower()
                if role in roles:
                    return str(t.get("value") or t.get("content") or "")
    return ""


def _string_fields(row: dict) -> dict[str, str]:
    """All non-empty string fields of a row (structural fallback works on this)."""
    return {k: v for k, v in row.items() if isinstance(v, str) and v.strip()}


def _extract_output(row: dict) -> tuple[str, str]:
    """(target text, how it was found). Layers: known keys -> sharegpt turns ->
    longest string field. Never returns '' while the row has any string field."""
    for k in _OUT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v, f"key:{k}"
    v = _from_turns(row, ("gpt", "assistant", "model", "bot"))
    if v.strip():
        return v, "turns:assistant"
    fields = _string_fields(row)
    if fields:
        # the target of an SFT row is almost always its longest text field
        k = max(fields, key=lambda kk: len(fields[kk]))
        return fields[k], f"inferred_longest:{k}"
    return "", "failed:no_text_fields"


def _extract_context(row: dict) -> tuple[str, str]:
    """(prompt-side text, how it was found). Layers: known keys -> sharegpt turns ->
    all string fields except the inferred output."""
    parts = []
    hits = []
    for k in _CTX_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
            hits.append(k)
    if parts:
        return "\n".join(parts), "keys:" + ",".join(hits)
    v = _from_turns(row, ("system", "human", "user"))
    if v.strip():
        return v, "turns:user"
    fields = _string_fields(row)
    if fields:
        out_k = max(fields, key=lambda kk: len(fields[kk]))
        rest = {k: v for k, v in fields.items() if k != out_k}
        if rest:
            return "\n".join(rest.values()), "inferred_rest"
        return "", "failed:only_output_field"  # single-field row (e.g. raw text): no context
    return "", "failed:no_text_fields"


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


def _domain_scores(rows: list[dict]) -> dict[str, float]:
    """Per-domain evidence fractions over the SAMPLED rows (not a one-row guess):
    function-calling = function-call-format fraction; math = rows whose output carries
    '#### ', \\boxed{} or 'ANSWER:'; medical/code = keyword-hit fractions. The label is
    argmax (>=0.3, else 'general') — scores stay visible so the leader sees borderline cases."""
    denom = max(len(rows), 1)
    fc = med = code = math = 0
    for r in rows:
        out, _ = _extract_output(r)
        ctx, _ = _extract_context(r)
        blob = (ctx + "\n" + out).lower()
        if _classify_output(out) == "function_call_json" or "available tools" in blob or "<tool" in out.lower():
            fc += 1
        if "#### " in out or re.search(r"\\boxed\{", out) or "answer:" in out.lower():
            math += 1
        if any(kw in blob for kw in _MEDICAL_KW):
            med += 1
        if any(m in out for m in _CODE_MARKERS):
            code += 1
    return {
        "function-calling": round(fc / denom, 3),
        "math": round(math / denom, 3),
        "medical": round(med / denom, 3),
        "code": round(code / denom, 3),
    }


def _domain_label(scores: dict[str, float]) -> str:
    label, score = max(scores.items(), key=lambda kv: kv[1])
    return label if score >= 0.3 else "general"


def profile_source(pool_root: str, source: str,
                   n_sample: int = 200, seed: int = 42) -> dict[str, Any]:
    try:
        ds = _load_split(pool_root, source)
    except Exception as e:  # noqa: BLE001 — a source we can't load is noted, not fatal
        return {"source": source, "n_rows": 0,
                "error": f"could not load: {type(e).__name__}: {e}"}
    n = len(ds)
    take = min(n_sample, n)
    try:
        sample = ds.shuffle(seed=seed).select(range(take))
    except Exception:  # noqa: BLE001
        sample = ds.select(range(take))
    rows = [dict(r) for r in sample]
    schema = sorted(rows[0].keys()) if rows else []

    fmt = {"function_call_json": 0, "natural_language": 0, "code": 0, "other": 0}
    neg = 0
    first_out = first_ctx = ""
    out_modes: dict[str, int] = {}
    empty_out = 0
    for r in rows:
        out, out_mode = _extract_output(r)
        ctx, _ = _extract_context(r)
        out_modes[out_mode] = out_modes.get(out_mode, 0) + 1
        if not out.strip():
            empty_out += 1
        if not first_out and out.strip():
            first_out, first_ctx = out, ctx
        fmt[_classify_output(out)] += 1
        if _is_negative(ctx, out):
            neg += 1
    denom = max(len(rows), 1)
    fmt_frac = {k: round(v / denom, 3) for k, v in fmt.items()}
    neg_ratio = round(neg / denom, 3)
    dom_scores = _domain_scores(rows)

    # No silent zeros: the extraction outcome is always explicit. If half or more of the
    # sampled rows yielded no target text, the profile is DEGRADED — downstream must not
    # read its stats as evidence (see the openmathinstruct2 incident: 13.9M rows reported
    # as "zero rows" because extraction silently missed the schema).
    degraded = rows and empty_out * 2 >= len(rows)
    raw0 = rows[0] if rows else {}
    sample_raw = {
        k: (v[:200] if isinstance(v, str) else v) for k, v in list(raw0.items())[:8]
    }

    # Raw statistics ONLY — no target_fit, no quality verdict, no usage hint. Format
    # fit against the bench is a fuzzy judgment: the investigator writes a per-source
    # textual note over these stats + samples (it holds the eval contract); the leader
    # judges from stats + note. A regex's 0.0 would only pretend precision.
    profile = {
        "source": source,
        "n_rows": n,
        "n_sampled": len(rows),
        "schema": schema,
        "domain": {"label": _domain_label(dom_scores), "scores": dom_scores},
        "sample": {"context": first_ctx[:200], "output": first_out[:200]},
        "sample_raw": sample_raw,
        "output_format": fmt_frac,
        "negatives": neg_ratio,
        "extraction": {"output_modes": out_modes, "empty_outputs": empty_out},
    }
    if degraded:
        profile["degraded"] = (
            f"{empty_out}/{len(rows)} sampled rows yielded no target text — stats are "
            "not evidence; re-probe before judging this source"
        )
    return profile


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
    ap.add_argument("--n-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.all_sources:
        profiles = [profile_source(args.pool_root, s, args.n_sample, args.seed)
                    for s in _iter_sources(args.pool_root)]
        json.dump(profiles, sys.stdout, ensure_ascii=False, indent=2)
    elif args.source:
        json.dump(profile_source(args.pool_root, args.source, args.n_sample, args.seed),
                  sys.stdout, ensure_ascii=False, indent=2)
    else:
        ap.error("give --source <id> or --all-sources")
    print()


if __name__ == "__main__":
    main()
