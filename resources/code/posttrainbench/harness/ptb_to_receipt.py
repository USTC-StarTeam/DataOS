#!/usr/bin/env python3
"""Convert a PostTrainBench evaluate.py flat-metrics JSON into a DataOS external receipt.

Usage: ptb_to_receipt.py <bench> <ptb_json> <receipt_json>

Calibrated against real outputs (2026-06-11, Qwen3-1.7B, limit 20):
- gsm8k emits exactly {"accuracy", "stderr"}.
- bfcl emits {"accuracy", "stderr", "exec_multiple_acc", "all", "exec_multiple_sterr", "all2"}
  where accuracy == exec_multiple_acc == all, and **"all2" is a stderr despite the name**.
- headline = "accuracy" for every bench (aime2025/gpqamain same inspect family; healthbench's
  rubric-weighted score and arenahardwriting's winrate are also exposed as "accuracy").
Metrics are prefixed "<bench>::" so suite-merging never collides across benches.
"""
import json
import sys


def flatten(payload, prefix=""):
    out = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):  # healthbench by_theme / by_axis
            out.update(flatten(value, name + "/"))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def main(bench, src, dst):
    raw = json.load(open(src))
    flat = flatten(raw)
    score = flat.get("accuracy")
    if score is None:  # last resort: first scalar that isn't an error bar / counter
        score = next(
            (v for k, v in flat.items()
             if "stderr" not in k and "sterr" not in k and k != "all2"
             and "n_" not in k and "calls" not in k),
            None,
        )
    if score is None:
        sys.exit(f"no usable metric in {src}: keys={sorted(raw)}")
    metrics = {f"{bench}::{key}": value for key, value in flat.items()}
    with open(dst, "w") as handle:
        json.dump({"score": score, "metrics": metrics}, handle, indent=2)
    print(f"[receipt] {bench} score={score} -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
