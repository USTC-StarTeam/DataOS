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
    parser.add_argument("--workers", type=int, default=None, help="healthbench per-rubric judge workers")
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_healthbench(task_mod, predictions: list[dict[str, Any]]) -> list[float]:
    # Reuse pre-main's exact rubric-weighted scoring (PostTrainBenchHealthBench.process_results
    # -> _health_criterion_met, ThreadPoolExecutor) by feeding each prediction back as the doc.
    task = task_mod.PostTrainBenchHealthBench.__new__(task_mod.PostTrainBenchHealthBench)
    scores: list[float] = []
    for index, pred in enumerate(predictions):
        doc = {
            "prompt": pred["prompt"],
            "rubrics": pred["rubrics"],
            "prompt_id": pred.get("prompt_id", index),
        }
        try:
            result = task.process_results(doc, [pred.get("answer", "")])
            value = float(result["healthbench_accuracy"])
        except Exception as exc:  # per-doc failure: log + skip, do not abort.
            print(
                f"[judge] healthbench doc {doc['prompt_id']} skipped: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            continue
        if math.isnan(value):
            print(f"[judge] healthbench doc {doc['prompt_id']} skipped: nan score", file=sys.stderr)
            continue
        scores.append(value)
    return scores


def score_arenahardwriting(task_mod, predictions: list[dict[str, Any]]) -> list[float]:
    # Reuse pre-main's exact pairwise scoring. process_results normally reads the baseline
    # from disk via _baseline_answer; here the baseline_answer is carried in predictions.jsonl,
    # so we call _judge_pair directly with the same two-game / inversion / mean logic.
    task = task_mod.PostTrainBenchArenaHardWriting.__new__(task_mod.PostTrainBenchArenaHardWriting)
    strip = task_mod._strip_thinking
    limit = task_mod._limit_repetitions
    scores: list[float] = []
    for index, pred in enumerate(predictions):
        doc = {
            "uid": pred.get("uid", index),
            "category": pred["category"],
            "prompt": pred["prompt"],
        }
        try:
            answer = limit(strip(pred.get("answer", "")))
            baseline = pred["baseline_answer"]
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
            continue
        if math.isnan(value):
            print(f"[judge] arena doc {doc['uid']} skipped: nan score", file=sys.stderr)
            continue
        scores.append(value)
    return scores


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
    if args.workers is not None:
        os.environ["POSTTRAINBENCH_HEALTHBENCH_CRITERIA_WORKERS"] = str(args.workers)
    # Judging must actually call the API here even though generation disabled it.
    os.environ.pop("POSTTRAINBENCH_JUDGE_DISABLED", None)

    task_mod = load_task_module()
    predictions = load_predictions(Path(args.predictions))

    if args.bench == "healthbench":
        scores = score_healthbench(task_mod, predictions)
    else:
        scores = score_arenahardwriting(task_mod, predictions)

    headline = float(np.mean(scores)) if scores else None
    score = headline * 100.0 if headline is not None else None
    patch_summary(Path(args.output), score, args.bench)
    print(json.dumps({"score": score, "bench": args.bench}))


if __name__ == "__main__":
    main()
