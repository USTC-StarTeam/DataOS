from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = [
    "Qwen3-1.7B-Base",
    "Qwen3-1.7B",
    "Qwen3-4B-Base",
    "Qwen3-4B",
    "SmolLM3-3B-Base",
    "SmolLM3-3B",
    "gemma-3-4b-pt",
    "gemma-3-4b-it",
]

DEFAULT_TASKS = [
    "gsm8k",
    "gpqamain",
    "bfcl",
    "humaneval",
    "arenahardwriting",
    "healthbench",
    "aime2025",
]

JUDGE_TASKS = {"arenahardwriting", "healthbench"}


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "matrix_events.jsonl"
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    models = args.models or DEFAULT_MODELS
    tasks = args.tasks or DEFAULT_TASKS

    jobs = queue.Queue()
    for model in models:
        for task in tasks:
            jobs.put((model, task))

    state_lock = threading.Lock()

    def write_event(event: dict[str, Any]) -> None:
        event = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with state_lock:
            with state_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def worker(gpu: str) -> None:
        while True:
            try:
                model, task = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                run_one(args, output_root, model, task, gpu, write_event)
            finally:
                jobs.task_done()

    write_event(
        {
            "event": "matrix_start",
            "gpus": gpus,
            "models": models,
            "tasks": tasks,
            "output_root": str(output_root),
        }
    )
    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_event({"event": "matrix_done"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PostTrainBench lm-eval model matrix.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-root", default="/home/hlv/dataos_eval_models")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-connections", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--runner-override", default=None)
    parser.add_argument("--posttrainbench-eval-src", default="/home/hlv/posttrainbench_eval_src")
    parser.add_argument("--benchmarks-dir", default="/home/hlv/posttrainbench_benchmarks")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_one(
    args: argparse.Namespace,
    output_root: Path,
    model: str,
    task: str,
    gpu: str,
    write_event,
) -> None:
    output_dir = output_root / model / task
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if not args.force and _completed(summary_path):
        write_event({"event": "skip_completed", "model": model, "task": task, "gpu": gpu})
        return

    env = _job_env(args, gpu)
    if task in JUDGE_TASKS and not _has_judge_credentials(env):
        blocked = {
            "task": task,
            "model_name": model,
            "returncode": None,
            "status": "blocked_missing_judge_credentials",
            "summary_path": str(summary_path),
        }
        summary_path.write_text(json.dumps(blocked, indent=2, sort_keys=True), encoding="utf-8")
        write_event({"event": "blocked_missing_judge_credentials", "model": model, "task": task})
        return

    model_path = str(Path(args.model_root) / model)
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "run_posttrainbench_protocol_eval.py"),
        "--task",
        task,
        "--model-path",
        model_path,
        "--model-name",
        model,
        "--output-dir",
        str(output_dir),
        "--python",
        args.python,
        "--limit",
        str(args.limit),
        "--batch-size",
        args.batch_size,
        "--posttrainbench-eval-src",
        args.posttrainbench_eval_src,
        "--benchmarks-dir",
        args.benchmarks_dir,
    ]
    if args.max_tokens is not None:
        command.extend(["--max-tokens", str(args.max_tokens)])
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.max_connections is not None:
        command.extend(["--max-connections", str(args.max_connections)])
    if args.gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.runner_override:
        command.extend(["--runner-override", args.runner_override])
    log_path = output_dir / "matrix_runner.log"
    write_event({"event": "start", "model": model, "task": task, "gpu": gpu})
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    write_event(
        {
            "event": "finish",
            "model": model,
            "task": task,
            "gpu": gpu,
            "returncode": proc.returncode,
            "elapsed_seconds": round(time.time() - started, 2),
            "output_dir": str(output_dir),
        }
    )


def _job_env(args: argparse.Namespace, gpu: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}",
            "HF_ENDPOINT": env.get("HF_ENDPOINT", "https://hf-mirror.com"),
            "HF_HOME": env.get("HF_HOME", "/home/hlv/dataos_eval_hf_cache"),
            "HF_DATASETS_CACHE": str(Path(args.benchmarks_dir) / "hf_datasets"),
            "VLLM_WORKER_MULTIPROC_METHOD": env.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
        }
    )
    _configure_judge_env(env)
    return env


def _configure_judge_env(env: dict[str, str]) -> None:
    judge_model = env.get("POSTTRAINBENCH_JUDGE_MODEL") or env.get("BAILIAN_JUDGE_MODEL")
    if not judge_model and (env.get("DASHSCOPE_API_KEY") or env.get("BAILIAN_API_KEY")):
        judge_model = "qwen3.5-plus"
    if judge_model and not env.get("POSTTRAINBENCH_JUDGE_MODEL"):
        env["POSTTRAINBENCH_JUDGE_MODEL"] = judge_model

    judge_api_key = (
        env.get("OPENAI_API_KEY")
        or env.get("POSTTRAINBENCH_JUDGE_API_KEY")
        or env.get("DASHSCOPE_API_KEY")
        or env.get("BAILIAN_API_KEY")
    )
    if judge_api_key and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = judge_api_key

    judge_base_url = (
        env.get("OPENAI_BASE_URL")
        or env.get("POSTTRAINBENCH_JUDGE_BASE_URL")
        or env.get("DASHSCOPE_BASE_URL")
    )
    if judge_base_url and not env.get("OPENAI_BASE_URL"):
        env["OPENAI_BASE_URL"] = judge_base_url
    elif (
        env.get("POSTTRAINBENCH_JUDGE_MODEL", "").lower().startswith("qwen")
        and not env.get("OPENAI_BASE_URL")
    ):
        env["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _has_judge_credentials(env: dict[str, str]) -> bool:
    return bool(env.get("OPENAI_API_KEY"))


def _completed(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return summary.get("returncode") == 0


if __name__ == "__main__":
    main()
