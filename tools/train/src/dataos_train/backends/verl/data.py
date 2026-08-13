"""Canonical RL records -> verl parquet. Runs inside the verl job.

Canonical RL record: ``{prompt, ground_truth, data_source}``. Prompts are taken AS-IS by
default — aligning them to the eval template is the driving agent's data-stage job. Pass
``--template-file`` (a file containing ``{prompt}``) to wrap at convert time instead.
``pyarrow`` is imported lazily (verl env only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataos_launch.io import atomic_write_text

from ...data.convert import read_records


def read_template(path: str | None) -> str:
    """Template wrapping ``{prompt}``; identity when no file is given."""
    if not path:
        return "{prompt}"
    return Path(path).read_text(encoding="utf-8")


def _ground_truth(rec: dict, bench: str) -> str:
    gt = rec.get("ground_truth")
    if gt is None:
        gt = rec.get("answer") or rec.get("target") or rec.get("output") or ""
    gt = str(gt)
    if bench.lower().startswith("gsm8k") and "####" in gt:
        gt = gt.rsplit("####", 1)[-1].strip()  # extract bare number, matching the verifier
    return gt


def to_verl_rows(records: list[dict], bench: str, template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        prompt = rec.get("prompt") or rec.get("question") or rec.get("instruction") or ""
        rows.append(
            {
                "data_source": rec.get("data_source", bench),
                "prompt": [{"role": "user", "content": template.format(prompt=prompt)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": _ground_truth(rec, bench)},
                "extra_info": {"index": i, "question": prompt},
            }
        )
    return rows


def write_parquet(rows: list[dict], out_dir: str) -> int:
    import pyarrow as pa  # lazy: verl env only
    import pyarrow.parquet as pq

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), str(d / "train.parquet"))
    val = rows[: min(64, len(rows))] or rows
    pq.write_table(pa.Table.from_pylist(val), str(d / "val.parquet"))
    atomic_write_text(d / "_count.json", json.dumps({"rows": len(rows)}))
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataos_train.backends.verl.data")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--informat", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--template-file", dest="template_file", default=None)
    args = ap.parse_args(argv)
    records = read_records(args.inp, args.informat)
    rows = to_verl_rows(records, args.bench, read_template(args.template_file))
    n = write_parquet(rows, args.out)
    print(json.dumps({"rows": n, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
