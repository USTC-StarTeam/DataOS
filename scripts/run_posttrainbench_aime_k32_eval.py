from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inspect_ai import Epochs
from inspect_ai import eval as inspect_eval
from inspect_ai.util._display import init_display_type

import inspect_evals.aime2025


def main() -> None:
    args = parse_args()
    init_display_type("plain")

    other_kwargs: dict[str, Any] = {}
    if args.limit is not None and args.limit != -1:
        other_kwargs["limit"] = args.limit

    model_args = {"gpu_memory_utilization": args.gpu_memory_utilization}
    model_args.update(_template_kwargs(args.model_path, args.templates_dir))

    generate_kwargs: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    top_k = args.top_k
    if top_k is None:
        top_k = 20 if "qwen" in args.model_path.lower() else -1
    if top_k >= 0:
        generate_kwargs["top_k"] = top_k

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_out = inspect_eval(
        inspect_evals.aime2025.aime2025(),
        model=f"vllm/{args.model_path}",
        model_args=model_args,
        epochs=Epochs(args.epochs, reducer="mean"),
        score_display=False,
        timeout=18000000,
        attempt_timeout=18000000,
        log_realtime=False,
        log_format="json",
        log_dir=str(log_dir),
        max_connections=args.max_connections,
        **generate_kwargs,
        **other_kwargs,
    )

    if len(eval_out) != 1:
        raise RuntimeError(f"Expected one eval output, got {len(eval_out)}")
    if len(eval_out[0].results.scores) != 1:
        raise RuntimeError(f"Expected one score entry, got {eval_out[0].results.scores}")

    metrics: dict[str, Any] = {
        "model": args.model_path,
        "epochs": args.epochs,
        "max_tokens": args.max_tokens,
        "max_connections": args.max_connections,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": top_k,
    }
    for key, value in eval_out[0].results.scores[0].metrics.items():
        metrics[key] = value.value
    metrics.update(_raw_accuracy(getattr(eval_out[0], "samples", None)))

    Path(args.json_output_file).write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PostTrainBench AIME 2025 k32 protocol.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--templates-dir", required=True)
    parser.add_argument("--json-output-file", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--max-connections", type=int, default=6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def _model_type(model_path: str) -> str:
    lower = model_path.lower()
    if "qwen" in lower:
        return "qwen"
    if "llama" in lower:
        return "llama"
    if "gemma" in lower:
        return "gemma"
    if "smollm" in lower:
        return "smollm"

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    architecture = config["architectures"][0].lower()
    if "gemma" in architecture:
        return "gemma"
    if "llama" in architecture:
        return "llama"
    if "qwen" in architecture:
        return "qwen"
    if "smollm" in architecture:
        return "smollm"
    raise ValueError(architecture)


def _template_kwargs(model_path: str, templates_dir: str) -> dict[str, str]:
    model_type = _model_type(model_path)
    if model_type == "qwen":
        template = "qwen3.jinja"
    elif model_type == "llama":
        template = "llama3.jinja"
    elif model_type == "gemma":
        template = "gemma3.jinja"
    elif model_type == "smollm":
        template = "smollm.jinja"
    else:
        raise ValueError(model_type)
    return {"chat_template": os.path.join(templates_dir, template)}


def _score_value(sample: Any) -> Any:
    scores = getattr(sample, "scores", None)
    if scores is None and isinstance(sample, dict):
        scores = sample.get("scores")
    if not isinstance(scores, dict):
        return None
    for score in scores.values():
        if isinstance(score, dict):
            return score.get("value")
        return getattr(score, "value", None)
    return None


def _is_correct(value: Any) -> bool:
    return value in ("C", "correct", True, 1, 1.0, "1", "1.0")


def _raw_accuracy(samples: list[Any] | None) -> dict[str, Any]:
    if not samples:
        return {}
    values = [_score_value(sample) for sample in samples]
    values = [value for value in values if value is not None]
    if not values:
        return {}
    if all(isinstance(value, (int, float, bool)) for value in values):
        score_mean = sum(float(value) for value in values) / len(values)
        return {"sample_score_total": len(values), "sample_score_mean": score_mean}
    correct = sum(1 for value in values if _is_correct(value))
    total = len(values)
    acc = correct / total if total else math.nan
    stderr = math.sqrt(acc * (1.0 - acc) / total) if total else math.nan
    return {
        "raw_correct": correct,
        "raw_total": total,
        "raw_accuracy": acc,
        "raw_stderr": stderr,
    }


if __name__ == "__main__":
    main()

