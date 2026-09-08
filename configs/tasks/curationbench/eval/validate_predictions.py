"""Check full prediction coverage using the frozen evaluator's dataset definitions."""

from __future__ import annotations

import argparse
from pathlib import Path


def validate(results_dir: Path, model: str, benchmarks: list[str]) -> None:
    import pandas as pd
    from vlmeval.dataset import build_dataset

    for name in benchmarks:
        path = results_dir / model / f"{model}_{name}.xlsx"
        if not path.is_file():
            # Upstream infer-only exits each dataset before creating the top-level symlinks.
            candidates = list((results_dir / model).glob(f"T*/{model}_{name}.xlsx"))
            if not candidates:
                raise ValueError(f"missing benchmark predictions: {path}")
            path = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        predictions = pd.read_excel(path, keep_default_na=False)
        dataset = build_dataset(name)
        if dataset is None:
            raise ValueError(f"unknown benchmark: {name}")
        if not {"index", "prediction"} <= set(predictions.columns):
            raise ValueError(f"missing prediction columns: {path}")
        expected = set(dataset.data["index"].astype(str))
        actual = predictions["index"].astype(str)
        if len(predictions) != len(expected) or set(actual) != expected:
            raise ValueError(f"incomplete or duplicate prediction indices: {path}")
        values = predictions["prediction"]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"empty predictions: {path}")
        if values.astype(str).str.contains("Failed to obtain answer via API", regex=False).any():
            raise ValueError(f"failed predictions: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    args = parser.parse_args()
    validate(args.results_dir, args.model, args.benchmarks)


if __name__ == "__main__":
    main()
