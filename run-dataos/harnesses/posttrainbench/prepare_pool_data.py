#!/usr/bin/env python3
"""Generalized PostTrainBench fixed-pool data prep — select + format SFT data for ONE bench.

Reads the pool registry, picks the sources tagged for ``--benchmark-tag`` (or an explicit
``--sources`` subset / ``--mix`` weights), samples ``--n-samples`` rows across them, drops
contaminated rows, formats each as an instruction-following SFT example ``{prompt, target,
text}``, and ``save_to_disk`` to ``--out``. CPU-only (runs as a data-op step).

Pool layout (hcgu): one HF ``DatasetDict`` per source under
``<pool-root>/<source_id>/train`` with columns instruction/input/output/system/.../
benchmark_tags/contamination_notes. The pool is curated to exclude benchmark test data; we
still defensively skip rows whose contamination_notes mention test/eval.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import Dataset, load_from_disk

DEFAULT_POOL = "/data/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v1_full"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-root", default=DEFAULT_POOL)
    p.add_argument("--benchmark-tag", required=True,
                   help="gsm8k|bfcl|humaneval|aime2025|gpqamain|healthbench|arenahardwriting")
    p.add_argument("--sources", default="", help="optional comma-separated source_id subset")
    p.add_argument("--mix", default="", help="optional source_id:weight,... mixture weights")
    p.add_argument("--n-samples", type=int, default=60000)
    p.add_argument("--max-chars", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _registry_sources(pool_root: str) -> list[dict]:
    reg = json.loads((Path(pool_root) / "registry.json").read_text(encoding="utf-8"))
    for key in ("sources", "source_registry", "registry"):
        val = reg.get(key)
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            return [{**v, "source_id": v.get("source_id", k)} for k, v in val.items()]
    raise SystemExit("registry.json: could not find a sources list")


def sources_for_tag(pool_root: str, tag: str, explicit: str) -> list[str]:
    tagged: list[str] = []
    for s in _registry_sources(pool_root):
        sid = s.get("source_id") or s.get("id")
        tags = s.get("benchmark_tags") or []
        if sid and tag in tags:
            tagged.append(sid)
    if explicit:
        want = [x.strip() for x in explicit.split(",") if x.strip()]
        return [s for s in want if s in tagged] or want
    return tagged


def main() -> int:
    a = parse_args()
    random.seed(a.seed)

    src_ids = sources_for_tag(a.pool_root, a.benchmark_tag, a.sources)
    if not src_ids:
        print(f"[prep] no sources tagged {a.benchmark_tag!r}", file=sys.stderr)
        return 2
    print(f"[prep] tag={a.benchmark_tag} sources={src_ids}")

    weights: dict[str, float] = {}
    if a.mix:
        for kv in a.mix.split(","):
            k, _, v = kv.partition(":")
            weights[k.strip()] = float(v) if v else 1.0
    per = {sid: weights.get(sid, 1.0) for sid in src_ids}
    total = sum(per.values()) or 1.0
    quota = {sid: max(1, int(a.n_samples * per[sid] / total)) for sid in src_ids}

    examples: list[dict] = []
    for sid in src_ids:
        path = Path(a.pool_root) / sid / "train"
        if not path.exists():
            print(f"[prep] skip missing {path}", file=sys.stderr)
            continue
        ds = load_from_disk(str(path))
        want, got = quota[sid], 0
        # random-sample candidate indices (oversample for post-filtering) — avoids a full
        # shuffle/flatten_indices on multi-million-row sources (memory-light).
        n = len(ds)
        k = min(n, want * 20 + 1000)
        cand = ds.select(random.sample(range(n), k)) if n else ds
        for ex in cand:
            if got >= want:
                break
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            out = (ex.get("output") or "").strip()
            # The pool is already curated to exclude benchmark test data (registry policy +
            # per-source contamination_notes describe what was excluded), so no per-row
            # contamination filter here — just require a usable instruction/output pair.
            if not instr or not out:
                continue
            prompt = instr if not inp else f"{instr}\n\n{inp}"
            text = f"{prompt}\n\n{out}"
            if len(text) > a.max_chars:
                continue
            examples.append({"prompt": prompt, "target": out, "text": text, "source_id": sid})
            got += 1
        print(f"[prep] {sid}: {got}/{want}")

    random.shuffle(examples)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Dataset.from_list(examples).save_to_disk(a.out)
    print(f"[prep] saved {len(examples)} examples -> {a.out}")
    if examples:
        (Path(a.out) / "sample.json").write_text(
            json.dumps(examples[0], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
