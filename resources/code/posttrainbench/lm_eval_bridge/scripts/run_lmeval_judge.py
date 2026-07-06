#!/usr/bin/env python3
"""Login-node LLM-as-judge scorer for PostTrainBench healthbench / arenahardwriting.

The compute node generates answers with the judge disabled
(POSTTRAINBENCH_JUDGE_DISABLED=1) and writes a predictions.jsonl. This script runs
on the H100 login node (which has internet) and reuses pre-main's exact judge
scoring from the ported task.py, so the numbers stay comparable to baselines.

It does NOT reimplement scoring:
  - healthbench: PostTrainBenchHealthBench.process_results (rubric-weighted, via
    _health_criterion_met) per prediction.
  - arenahardwriting: PostTrainBenchArenaHardWriting._judge_pair, both directions
    with the same baseline_first inversion and mean as process_results.

Env vars the caller must set on the login node:
  OPENAI_API_KEY, OPENAI_BASE_URL  (Bailian / dashscope compatible-mode)
  POSTTRAINBENCH_JUDGE_MODEL       (e.g. qwen3.5-plus; overridable via --judge-model)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

TASK_PY = (
    Path(__file__).resolve().parent.parent
    / "dataos/eval_tasks/posttrainbench/task.py"
)


def load_task_module():
    task_dir = TASK_PY.parent
    for value in (str(task_dir), str(task_dir.parent), str(task_dir.parent.parent)):
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("posttrainbench_judge_task", TASK_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import task module from {TASK_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Login-node PostTrainBench judge scorer.")
    parser.add_argument("--bench", required=True, choices=["healthbench", "arenahardwriting"])
    parser.add_argument("--predictions", required=True, help="predictions.jsonl written by the generation runner")
    parser.add_argument("--output", required=True, help="summary.json to update or create")
    parser.add_argument("--judge-model", default=None, help="overrides POSTTRAINBENCH_JUDGE_MODEL")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="judge document concurrency — predictions scored in parallel (default 32; "
        "chatanywhere handles ~32-64 concurrent, throughput ~45 calls/min).",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parallel_doc_scores(predictions, score_one, workers: int) -> list[float]:
    """Score each prediction independently and CONCURRENTLY (the predictions are the real
    bottleneck — each is many judge API calls), preserving order; per-doc failures return
    None and are dropped. score_one(index, pred) -> float | None. The per-doc scoring keeps
    its criteria serial (we parallelize at the doc level), so total in-flight ≈ workers."""
    results: list[float | None] = [None] * len(predictions)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(score_one, i, p): i for i, p in enumerate(predictions)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [v for v in results if v is not None]


def score_healthbench(task_mod, predictions: list[dict[str, Any]], workers: int) -> list[float]:
    # Reuse pre-main's exact rubric-weighted scoring (PostTrainBenchHealthBench.process_results
    # -> _health_criterion_met) by feeding each prediction back as the doc — scored per-doc in parallel.
    task = task_mod.PostTrainBenchHealthBench.__new__(task_mod.PostTrainBenchHealthBench)

    def score_one(index, pred):
        doc = {
            "prompt": pred["prompt"],
            "rubrics": pred["rubrics"],
            "prompt_id": pred.get("prompt_id", index),
        }
        try:
            value = float(task.process_results(doc, [pred.get("answer", "")])["healthbench_accuracy"])
        except Exception as exc:  # per-doc failure: log + skip, do not abort.
            print(
                f"[judge] healthbench doc {doc['prompt_id']} skipped: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            return None
        if math.isnan(value):
            print(f"[judge] healthbench doc {doc['prompt_id']} skipped: nan score", file=sys.stderr)
            return None
        return value

    return _parallel_doc_scores(predictions, score_one, workers)


def score_arenahardwriting(task_mod, predictions: list[dict[str, Any]], workers: int) -> list[float]:
    # Reuse pre-main's exact pairwise scoring (two games / baseline_first inversion / mean),
    # scored per-doc in parallel. Baseline answers come from disk via the task's _baseline_answer;
    # download() resolves _data_dir (<POSTTRAINBENCH_BENCHMARKS_DIR>/local/arena-hard-v2.0).
    task = task_mod.PostTrainBenchArenaHardWriting.__new__(task_mod.PostTrainBenchArenaHardWriting)
    task.download()
    strip = task_mod._strip_thinking
    limit = task_mod._limit_repetitions

    def score_one(index, pred):
        doc = {
            "uid": pred.get("uid", index),
            "category": pred["category"],
            "prompt": pred["prompt"],
        }
        try:
            answer = limit(strip(pred.get("answer", "")))
            baseline = task._baseline_answer(doc)
            doc_scores: list[float] = []
            for score in task._judge_pair(doc, baseline, answer, baseline_first=True):
                doc_scores.append(1.0 - score)
            doc_scores.extend(task._judge_pair(doc, answer, baseline, baseline_first=False))
            value = float(np.mean(doc_scores)) if doc_scores else float("nan")
        except Exception as exc:  # per-doc failure: log + skip, do not abort.
            print(
                f"[judge] arena doc {doc['uid']} skipped: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            return None
        if math.isnan(value):
            print(f"[judge] arena doc {doc['uid']} skipped: nan score", file=sys.stderr)
            return None
        return value

    return _parallel_doc_scores(predictions, score_one, workers)


def patch_summary(output: Path, score: float | None, bench: str) -> None:
    summary: dict[str, Any] = {}
    if output.exists():
        try:
            summary = json.loads(output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = {}
    summary["score"] = score
    summary["needs_judge"] = False
    summary.setdefault("bench", bench)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.judge_model:
        os.environ["POSTTRAINBENCH_JUDGE_MODEL"] = args.judge_model
    # We parallelize at the DOC level (the predictions are the real bottleneck), so keep each
    # doc's per-criterion scoring serial — total in-flight judge calls then ≈ `workers`.
    # Floor judge concurrency at 16: a sub-agent once passed --workers 1, making a
    # HealthBench round take ~3h instead of ~10min. Default 32 when unset.
    workers = max(16, args.workers) if args.workers is not None else 32
    os.environ["POSTTRAINBENCH_HEALTHBENCH_CRITERIA_WORKERS"] = "1"
    # Judging must actually call the API here even though generation disabled it.
    os.environ.pop("POSTTRAINBENCH_JUDGE_DISABLED", None)

    task_mod = load_task_module()
    predictions = load_predictions(Path(args.predictions))

    if args.bench == "healthbench":
        scores = score_healthbench(task_mod, predictions, workers)
    else:
        scores = score_arenahardwriting(task_mod, predictions, workers)

    headline = float(np.mean(scores)) if scores else None
    score = headline * 100.0 if headline is not None else None
    patch_summary(Path(args.output), score, args.bench)
    print(json.dumps({"score": score, "bench": args.bench}))


if __name__ == "__main__":
    main()
