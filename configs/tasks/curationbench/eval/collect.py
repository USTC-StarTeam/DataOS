"""Convert Curation-Bench's pinned VLMEvalKit output to DataOS summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _rows(value: object) -> list[dict]:
    """Turn VLMEvalKit's DataFrame.to_dict() JSON shape back into rows."""
    if not isinstance(value, dict) or not value:
        return []
    columns = {str(key): item for key, item in value.items()}
    if not all(isinstance(item, dict) for item in columns.values()):
        return [columns]
    indices = []
    for column in columns.values():
        for index in column:
            if index not in indices:
                indices.append(index)
    return [
        {name: column.get(index) for name, column in columns.items()}
        for index in indices
    ]


def _score(benchmark: str, value: object, selector: dict) -> float:
    rows = _rows(value)
    match = selector.get("row") or {}
    selected = [
        row
        for row in rows
        if all(str(row.get(field)).lower() == str(expected).lower()
               for field, expected in match.items())
    ]
    if len(selected) != 1:
        raise ValueError(
            f"{benchmark} selector matched {len(selected)} rows; expected exactly one"
        )
    row = selected[0]
    if selector.get("mean_fields"):
        numbers = [_number(row.get(field)) for field in selector["mean_fields"]]
        if any(number is None for number in numbers):
            raise ValueError(f"{benchmark} is missing one of {selector['mean_fields']}")
        score = sum(numbers) / len(numbers)
    else:
        score = _number(row.get(selector["field"]))
        if score is None:
            raise ValueError(f"{benchmark} has no numeric {selector['field']!r} score")
    return score * float(selector.get("multiply", 1))


def collect(suite_dir: Path, output_dir: Path, config: dict) -> dict:
    state = json.loads((suite_dir / "suite_state.json").read_text(encoding="utf-8"))
    completed = [
        target
        for target in state.get("targets") or []
        if str(target.get("state", "")).lower() == "completed" and target.get("run_dir")
    ]
    if not completed:
        raise ValueError("suite has no completed target")
    target = completed[-1]
    task = yaml.safe_load(Path(target["task_path"]).read_text(encoding="utf-8"))["task"]
    benchmarks = list(task["target_evals"])
    results_path = Path(target["run_dir"]) / str(config["results_file"])
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if len(results) != 1:
        raise ValueError(f"expected one model in {results_path}, found {len(results)}")
    model_name, model_results = next(iter(results.items()))
    if not isinstance(model_results, dict):
        raise ValueError(f"model result in {results_path} must be a mapping")

    selectors = config["selectors"]
    maxima = config["normalization"]
    metrics: dict[str, float] = {}
    normalized: dict[str, float] = {}
    for benchmark in benchmarks:
        if benchmark not in selectors:
            raise ValueError(f"missing score selector for {benchmark}")
        score = _score(benchmark, model_results.get(benchmark), selectors[benchmark])
        maximum = float(maxima[benchmark])
        metrics[benchmark] = score
        normalized[benchmark] = score / maximum
    accuracy = sum(normalized.values()) / len(normalized)
    summary = {
        "score": accuracy,
        "accuracy": accuracy,
        "metrics": metrics,
        "normalized": normalized,
        "model_name": model_name,
        "task": target.get("task_id"),
        "native_results": str(results_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    try:
        summary = collect(args.suite_dir, args.output_dir, json.loads(args.config_json))
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports task data errors
        print(f"collector failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
