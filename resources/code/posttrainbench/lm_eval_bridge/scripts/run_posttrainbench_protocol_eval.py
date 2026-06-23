#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


TASK_ALIASES = {
    "gsm8k": "posttrainbench_gsm8k",
    "bfcl": "posttrainbench_bfcl",
    "gpqamain": "posttrainbench_gpqa_main",
    "humaneval": "posttrainbench_humaneval",
    "healthbench": "posttrainbench_healthbench",
    "aime2025": "posttrainbench_aime2025",
    "arenahardwriting": "posttrainbench_arena_hard_writing",
}

SCORE_KEYS = {
    "gsm8k": "accuracy,none",
    "bfcl": "bfcl_accuracy,none",
    "gpqamain": "accuracy,none",
    "humaneval": "pass@1,create_test",
    "healthbench": "healthbench_accuracy,none",
    "aime2025": "exact_match,k32",
    "arenahardwriting": "arena_winrate,none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility runner for PostTrainBench protocol evals.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--posttrainbench-eval-src", required=True)
    parser.add_argument("--benchmarks-dir", required=True)
    parser.add_argument("--inspect-pkgs", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def load_module(path: Path, name: str):
    task_dir = path.parent
    for value in (str(task_dir), str(task_dir.parent), str(task_dir.parent.parent)):
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metric_value(metric: Any) -> Any:
    return getattr(metric, "value", metric)


def extract_accuracy(metrics: dict[str, Any]) -> float:
    if "accuracy" in metrics:
        return float(metrics["accuracy"])
    for key in ("bfcl_accuracy", "pass@1", "exact_match"):
        if key in metrics:
            return float(metrics[key])
    for key, value in metrics.items():
        if "accuracy" in key or "pass@1" in key or "exact_match" in key:
            return float(value)
    raise KeyError(f"could not find accuracy in metric keys: {sorted(metrics)}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def protocol_payload(args: argparse.Namespace, command: list[str], requires_api_key: bool, runner: str) -> dict[str, Any]:
    env_keys = sorted(
        key
        for key in os.environ
        if key.startswith("POSTTRAINBENCH_")
        or key
        in {
            "HF_DATASETS_CACHE",
            "HF_ENDPOINT",
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "OPENAI_BASE_URL",
            "PYTHONPATH",
            "PYTHONUNBUFFERED",
            "VLLM_API_KEY",
            "VLLM_WORKER_MULTIPROC_METHOD",
        }
    )
    task = args.task
    max_tokens = 4000 if task in {"gsm8k", "humaneval"} else 16000
    if task in {"healthbench", "arenahardwriting"}:
        max_tokens = 16384
    return {
        "benchmark": "PostTrainBench",
        "benchmarks_dir": args.benchmarks_dir,
        "command": command,
        "env_keys": env_keys,
        "gpu_memory_utilization": {"gsm8k": 0.5, "humaneval": 0.3}.get(task, 0.8),
        "inspect_pkgs": args.inspect_pkgs,
        "limit": args.limit,
        "max_connections": {"gsm8k": 2, "bfcl": 1, "gpqamain": 6, "humaneval": 1, "aime2025": 2}.get(task, 1),
        "max_model_len": 32768,
        "max_tokens": max_tokens,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "posttrainbench_eval_src": args.posttrainbench_eval_src,
        "requires_api_key": requires_api_key,
        "runner": runner,
        "runner_override": "runtime_repair_shim",
        "task": task,
        "task_generation_kwargs": {"max_gen_toks": max_tokens},
        "vllm_generation_config": "vllm",
    }


def write_summary(
    args: argparse.Namespace,
    out_dir: Path,
    start: float,
    accuracy: float,
    metrics: dict[str, Any],
    returncode: int = 0,
) -> None:
    alias = TASK_ALIASES[args.task]
    metric_key = SCORE_KEYS[args.task]
    metrics_payload = {
        "accuracy": accuracy,
        "results": {
            alias: {
                "alias": alias,
                metric_key: accuracy,
                f"{metric_key.split(',')[0]}_stderr,{metric_key.split(',', 1)[1] if ',' in metric_key else 'none'}": "N/A",
            }
        },
    }
    metrics_payload.update({k: v for k, v in metrics.items() if k not in metrics_payload})
    metrics_path = out_dir / "metrics.json"
    summary_path = out_dir / "summary.json"
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    stdout_path.touch(exist_ok=True)
    stderr_path.touch(exist_ok=True)
    write_json(metrics_path, metrics_payload)
    write_json(
        summary_path,
        {
            "elapsed_seconds": time.time() - start,
            "metrics": metrics_payload,
            "metrics_path": str(metrics_path),
            "model_name": args.model_name,
            "protocol_path": str(out_dir / "protocol.json"),
            "returncode": returncode,
            "score": accuracy * 100.0,
            "stderr_path": str(stderr_path),
            "stdout_path": str(stdout_path),
            "task": args.task,
        },
    )
    print(summary_path.read_text(encoding="utf-8"))


def prepare_env(args: argparse.Namespace) -> Path:
    eval_src = Path(args.posttrainbench_eval_src)
    os.environ["POSTTRAINBENCH_BENCHMARKS_DIR"] = args.benchmarks_dir
    os.environ["POSTTRAINBENCH_EVAL_SRC"] = str(eval_src)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    conda_bin = Path(sys.executable).resolve().parent
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(conda_bin) not in path_parts:
        os.environ["PATH"] = str(conda_bin) + os.pathsep + os.environ.get("PATH", "")
    for value in (args.inspect_pkgs, str(eval_src), str(eval_src / "tasks")):
        if value and value not in sys.path:
            sys.path.insert(0, value)
    return eval_src


def run_inspect_task(args: argparse.Namespace, out_dir: Path, eval_src: Path) -> tuple[float, dict[str, Any]]:
    from inspect_ai import eval as inspect_eval
    from inspect_ai.util._display import init_display_type

    init_display_type("plain")
    templates_dir = str(eval_src / "templates")
    task = args.task
    namespace = SimpleNamespace(model_path=args.model_path, templates_dir=templates_dir)
    limit_kwargs = {} if args.limit == -1 else {"limit": args.limit}
    log_dir = str(out_dir / "artifacts")

    if task == "gsm8k":
        mod = load_module(eval_src / "tasks/gsm8k/evaluate.py", "posttrainbench_gsm8k_eval")
        eval_task = mod.gsm8k_task()
        model_args = {"gpu_memory_utilization": 0.5, "max_model_len": 32768}
        model_args.update(mod.template_kwargs(namespace))
        eval_out = inspect_eval(
            eval_task,
            model=f"vllm/{args.model_path}",
            model_args=model_args,
            max_tokens=4000,
            max_connections=2,
            score_display=False,
            log_realtime=False,
            log_format="json",
            log_dir=log_dir,
            timeout=18000000,
            attempt_timeout=18000000,
            **limit_kwargs,
        )
    elif task == "bfcl":
        mod = load_module(eval_src / "tasks/bfcl/evaluate.py", "posttrainbench_bfcl_eval")
        eval_task = mod.bfcl_task()
        model_args = {
            "enable_auto_tool_choice": None,
            "tool_call_parser": mod.tool_call_parser_name(namespace),
            "gpu_memory_utilization": 0.8,
            "max_model_len": 32768,
        }
        model_args.update(mod.template_kwargs(namespace))
        eval_out = inspect_eval(
            eval_task,
            model=f"vllm/{args.model_path}",
            model_args=model_args,
            max_tokens=16000,
            max_connections=1,
            score_display=False,
            log_format="json",
            log_dir=log_dir,
            timeout=18000000,
            attempt_timeout=18000000,
            **limit_kwargs,
        )
    elif task == "gpqamain":
        mod = load_module(eval_src / "tasks/gpqamain/evaluate.py", "posttrainbench_gpqa_eval")
        eval_task = mod.gpqa_main()
        model_args = {"gpu_memory_utilization": 0.8, "max_model_len": 32768}
        model_args.update(mod.template_kwargs(namespace))
        eval_out = inspect_eval(
            eval_task,
            model=f"vllm/{args.model_path}",
            model_args=model_args,
            max_tokens=16000,
            max_connections=6,
            score_display=False,
            log_realtime=False,
            log_format="json",
            log_dir=log_dir,
            timeout=18000000,
            attempt_timeout=18000000,
            **limit_kwargs,
        )
    elif task == "humaneval":
        mod = load_module(eval_src / "tasks/humaneval/evaluate.py", "posttrainbench_humaneval_eval")
        eval_task = mod.inspect_evals.humaneval.humaneval(sandbox="local")
        model_args = {"gpu_memory_utilization": 0.3, "max_model_len": 32768}
        model_args.update(mod.template_kwargs(namespace))
        eval_out = inspect_eval(
            eval_task,
            model=f"vllm/{args.model_path}",
            model_args=model_args,
            max_tokens=4000,
            max_connections=1,
            score_display=False,
            log_realtime=False,
            log_format="json",
            log_dir=log_dir,
            timeout=18000000,
            attempt_timeout=18000000,
            **limit_kwargs,
        )
    elif task == "aime2025":
        mod = load_module(eval_src / "tasks/aime2025/evaluate.py", "posttrainbench_aime_eval")
        model_args = {"gpu_memory_utilization": 0.8, "max_model_len": 32768}
        model_args.update(mod.template_kwargs(namespace))
        eval_task = mod.inspect_evals.aime2025.aime2025()
        eval_out = inspect_eval(
            eval_task,
            model=f"vllm/{args.model_path}",
            model_args=model_args,
            max_tokens=16000,
            max_connections=2,
            epochs=32,
            temperature=0.6,
            top_p=0.95,
            score_display=False,
            log_realtime=False,
            log_format="json",
            log_dir=log_dir,
            timeout=18000000,
            attempt_timeout=18000000,
            **limit_kwargs,
        )
    else:
        raise ValueError(task)

    if len(eval_out) != 1 or len(eval_out[0].results.scores) != 1:
        raise RuntimeError(f"unexpected inspect result shape for {task}")
    metrics = {k: metric_value(v) for k, v in eval_out[0].results.scores[0].metrics.items()}
    accuracy = extract_accuracy(metrics)
    return accuracy, metrics


def run_healthbench_prediction(args: argparse.Namespace, out_dir: Path, eval_src: Path) -> tuple[float, dict[str, Any]]:
    task_dir = eval_src / "tasks/healthbench"
    mod = load_module(task_dir / "evaluate.py", "posttrainbench_healthbench_eval")
    templates_dir = str(eval_src / "templates")
    namespace = SimpleNamespace(
        model_path=args.model_path,
        max_new_tokens=16384,
        limit=args.limit,
        templates_dir=templates_dir,
    )
    examples = mod.load_healthbench()
    random.Random(42).shuffle(examples)
    if args.limit != -1:
        examples = examples[: args.limit]
    responses = mod.generate_answers(namespace, examples)
    prediction_path = Path(os.environ.get("POSTTRAINBENCH_HEALTHBENCH_PREDICTIONS", out_dir / "healthbench_predictions.jsonl"))
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for example, response in zip(examples, responses):
            rubrics = [
                {"criterion": item.criterion, "points": item.points, "tags": list(item.tags)}
                for item in example.rubrics
            ]
            handle.write(
                json.dumps(
                    {
                        "prompt_id": example.prompt_id,
                        "prompt": example.prompt,
                        "answer": response,
                        "rubrics": rubrics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return 0.0, {"accuracy": 0.0, "prediction_path": str(prediction_path), "records": len(examples)}


def run_arena_prediction(args: argparse.Namespace, out_dir: Path, eval_src: Path) -> tuple[float, dict[str, Any]]:
    task_dir = eval_src / "tasks/arenahardwriting"
    mod = load_module(task_dir / "evaluate.py", "posttrainbench_arena_eval")
    templates_dir = str(eval_src / "templates")
    model_alias = Path(args.model_path).name if Path(args.model_path).exists() else args.model_path.split("/")[-1]
    namespace = SimpleNamespace(
        model_path=args.model_path,
        model_alias=model_alias,
        max_new_tokens=16384,
        limit=args.limit,
        templates_dir=templates_dir,
        store_outputs=False,
    )
    old_cwd = os.getcwd()
    os.chdir(task_dir)
    try:
        _, answers = mod.generate_answers(namespace)
        questions = {item["uid"]: item for item in mod.get_questions(namespace)}
        baseline_answers = mod.load_model_answers(str(mod.DATA_PATH / "model_answer"))
    finally:
        os.chdir(old_cwd)
    prediction_path = Path(os.environ.get("POSTTRAINBENCH_ARENA_PREDICTIONS", out_dir / "arena_predictions.jsonl"))
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for uid, candidate in answers.items():
            question = questions[uid]
            category = question["category"]
            baseline_model = mod.JUDGE_SETTINGS[category]["baseline"]
            baseline = baseline_answers[baseline_model][uid]
            candidate_answer = candidate["messages"][-1]["content"]["answer"]
            baseline_answer = baseline["messages"][-1]["content"]["answer"]
            handle.write(
                json.dumps(
                    {
                        "uid": uid,
                        "category": category,
                        "prompt": question["prompt"],
                        "answer": candidate_answer,
                        "baseline_answer": baseline_answer,
                        "baseline_model": baseline.get("model", baseline_model),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return 0.0, {"accuracy": 0.0, "prediction_path": str(prediction_path), "records": len(answers)}


def main() -> None:
    args = parse_args()
    start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_src = prepare_env(args)

    command = [args.python, __file__, "--task", args.task, "--model-path", args.model_path, "--model-name", args.model_name]
    requires_api_key = args.task in {"healthbench", "arenahardwriting"}
    runner = "dataos_lmeval_native"
    write_json(out_dir / "protocol.json", protocol_payload(args, command, requires_api_key=requires_api_key, runner=runner))

    if args.task == "healthbench":
        accuracy, metrics = run_healthbench_prediction(args, out_dir, eval_src)
    elif args.task == "arenahardwriting":
        accuracy, metrics = run_arena_prediction(args, out_dir, eval_src)
    else:
        accuracy, metrics = run_inspect_task(args, out_dir, eval_src)
    write_summary(args, out_dir, start, accuracy, metrics)


if __name__ == "__main__":
    main()
