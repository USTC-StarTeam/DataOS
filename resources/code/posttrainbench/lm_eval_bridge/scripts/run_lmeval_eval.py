#!/usr/bin/env python3
"""Standalone PostTrainBench lm-evaluation-harness runner for one benchmark.

This reproduces the ``pre-main`` PostTrainBench lm-eval protocol so the agent
sub-agent can call it on an H100 Slurm node (conda env ``lm-eval``). It runs ONE
bench at a time through ``lm_eval.simple_evaluate`` and writes
``<output-dir>/summary.json`` with a native percentage ``score`` (0-100).

Per-bench serving (copied from pre-main):

* ``gpqamain``/``humaneval``: lm-eval's in-process vLLM backend (``model="vllm"``).
* ``gsm8k``/``bfcl``/``aime2025``/``healthbench``/``arenahardwriting``: an
  OpenAI-compatible ``vllm serve`` subprocess plus the registered lm-eval model
  ``dataos-local-tool-chat-completions``.

The judge benches (healthbench, arenahardwriting) GENERATE ONLY here -- the H100
compute nodes have no internet, so the judge runs later on the login node. They
are evaluated with ``predict_only=True`` (no ``process_results`` -> no judge
call); the generated answers are dumped to ``<output-dir>/predictions.jsonl`` and
``summary.json`` is written with ``score=null`` and ``needs_judge=true``.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# .../lm_eval_bridge/scripts/run_lmeval_eval.py -> package root is the parent of scripts/
BRIDGE_ROOT = SCRIPT_DIR.parent
POSTTRAINBENCH_DIR = BRIDGE_ROOT / "dataos" / "eval_tasks" / "posttrainbench"
TEMPLATES_DIR = POSTTRAINBENCH_DIR / "templates"

TASK_ALIASES = {
    "gsm8k": "posttrainbench_gsm8k",
    "humaneval": "posttrainbench_humaneval",
    "bfcl": "posttrainbench_bfcl",
    "gpqamain": "posttrainbench_gpqa_main",
    "aime2025": "posttrainbench_aime2025",
    "healthbench": "posttrainbench_healthbench",
    "arenahardwriting": "posttrainbench_arena_hard_writing",
}

# Benches served by lm-eval's in-process vLLM backend.
DIRECT_VLLM_TASKS = {"gpqamain", "humaneval"}
# Benches served by an OpenAI-compatible ``vllm serve`` subprocess.
SERVER_TASKS = {"gsm8k", "bfcl", "aime2025", "healthbench", "arenahardwriting"}
# Benches whose score is produced by an external judge (generate-only here).
JUDGE_TASKS = {"healthbench", "arenahardwriting"}

# vLLM GPU memory utilisation for the in-process vLLM backend.
DIRECT_GPU_MEMORY = {"humaneval": 0.3, "gpqamain": 0.8}

# Score metric per task, by name only; the lm-eval filter suffix is resolved at
# extraction time. (gsm8k accuracy, gpqamain accuracy, humaneval pass@1,
# bfcl bfcl_accuracy, aime2025 exact_match.)
SCORE_METRIC = {
    "gsm8k": "accuracy",
    "gpqamain": "accuracy",
    "humaneval": "pass@1",
    "bfcl": "bfcl_accuracy",
    "aime2025": "exact_match",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one PostTrainBench benchmark through lm-evaluation-harness."
    )
    parser.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="posttrainbench-model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Number of docs to evaluate. -1 means the full set.",
    )
    parser.add_argument(
        "--include-path",
        default=str(POSTTRAINBENCH_DIR),
        help="lm-eval TaskManager include_path (the posttrainbench task package).",
    )
    parser.add_argument(
        "--templates-dir",
        default=str(TEMPLATES_DIR),
        help="Directory containing the PostTrainBench chat-template jinja files.",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=None,
        help="Server-backend concurrency (how many eval requests vLLM batches together); "
        "default 32. Not score-affecting — raise for faster eval, but long-output benches "
        "(bfcl/healthbench, 16k tokens) are KV-cache bound, so very high values just queue.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Chat-template resolution (by model name/path substring)
# ---------------------------------------------------------------------------


def resolve_chat_template(task: str, model_name: str, model_path: str, templates_dir: Path) -> Path:
    lowered = f"{model_name} {model_path}".lower()
    if "qwen" in lowered:
        name = "qwen3.jinja"
    elif "smollm" in lowered:
        name = "smollm.jinja"
    elif "gemma" in lowered:
        name = "gemma3_tool_calling.jinja" if task == "bfcl" else "gemma3.jinja"
    else:
        raise ValueError(
            f"Cannot auto-select PostTrainBench chat template for model: {model_name} / {model_path}"
        )
    template = templates_dir / name
    if not template.exists():
        raise FileNotFoundError(f"Chat template does not exist: {template}")
    return template.resolve()


def tool_call_parser_name(model_path: str, model_name: str) -> str:
    lowered = f"{model_name} {model_path}".lower()
    if any(family in lowered for family in ("qwen", "gemma", "smollm")):
        return "hermes"
    if "llama" in lowered:
        return "llama3_json"
    return "hermes"


# ---------------------------------------------------------------------------
# gen_kwargs per bench (copied from pre-main)
# ---------------------------------------------------------------------------


def generation_kwargs(task: str) -> dict[str, Any]:
    if task == "gsm8k":
        return {"max_gen_toks": 4000}
    if task == "humaneval":
        return {"max_gen_toks": 4000, "until": ["</s>", "<|im_end|>", "<|eot_id|>"]}
    if task == "gpqamain":
        return {"max_gen_toks": 16000, "temperature": 1.0}
    if task == "bfcl":
        return {
            "max_gen_toks": 16000,
            "temperature": 0.6,
            "top_p": 0.95,
            "until": ["<|im_end|>", "<|eot_id|>"],
        }
    if task == "aime2025":
        return {
            "max_gen_toks": 16000,
            "temperature": 0.6,
            "top_p": 0.95,
            "dataos_omit_seed": True,
        }
    if task in {"healthbench", "arenahardwriting"}:
        return {"max_gen_toks": 16384, "dataos_official_chat_payload": True}
    raise ValueError(task)


# ---------------------------------------------------------------------------
# OpenAI-compatible chat backend defaults (copied from pre-main)
# ---------------------------------------------------------------------------


# Server-backend concurrency: how many eval requests vLLM batches together. Uniform
# default (the per-bench 1/2/6 inherited from pre-main left bfcl/healthbench serial);
# override per run with --num-concurrent. Not score-affecting — raising it only changes
# batching, not generation — so it is safe to tune for speed.
DEFAULT_NUM_CONCURRENT = 32


def default_timeout(task: str) -> int:
    if task == "aime2025":
        return 18_000_000
    if task in {"arenahardwriting", "bfcl", "gpqamain", "healthbench"}:
        return 1800
    if task == "humaneval":
        return 900
    return 300


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def configure_env(task: str, output_dir: Path) -> None:
    os.environ.setdefault("VLLM_API_KEY", "inspectai")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if task == "humaneval":
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    if task == "gpqamain":
        os.environ.setdefault("POSTTRAINBENCH_GPQA_SHUFFLE_SEED", "42")
    if task == "aime2025":
        # The OpenAI-compatible backend bounds its async request window by this.
        os.environ.setdefault("DATAOS_LMEVAL_API_WINDOW", "32")
        os.environ.setdefault(
            "POSTTRAINBENCH_AIME_EXTRACTION_LOG",
            str(output_dir / "aime_extraction_events.jsonl"),
        )
    if task in JUDGE_TASKS:
        # Compute nodes have no internet: forbid any judge API call. We also run
        # judge benches with predict_only=True so process_results never fires.
        os.environ["POSTTRAINBENCH_JUDGE_DISABLED"] = "1"


def register_tool_chat_model() -> None:
    """Import the openai_tool_chat module so its lm-eval model registers."""
    module_path = BRIDGE_ROOT / "dataos" / "eval_models" / "openai_tool_chat.py"
    spec = importlib.util.spec_from_file_location(
        "dataos_eval_models_openai_tool_chat", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load openai_tool_chat from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


# ---------------------------------------------------------------------------
# vLLM OpenAI-compatible server subprocess
# ---------------------------------------------------------------------------


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_vllm_server(
    task: str,
    model_path: str,
    chat_template: Path,
    model_name: str,
    output_dir: Path,
) -> tuple[subprocess.Popen, int]:
    port = find_free_port()
    vllm_bin = Path(sys.executable).with_name("vllm")
    command = [
        str(vllm_bin) if vllm_bin.exists() else "vllm",
        "serve",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--api-key",
        os.environ.get("VLLM_API_KEY", "inspectai"),
        "--chat-template",
        str(chat_template),
        "--generation-config",
        "vllm",
    ]
    if task == "bfcl":
        command.extend(
            [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                tool_call_parser_name(model_path, model_name),
            ]
        )
    log_path = output_dir / "vllm_server.log"
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        wait_for_health(port, process)
    except Exception:
        if process.poll() is None:
            process.kill()
        raise
    return process, port


def wait_for_health(port: int, process: subprocess.Popen, timeout: int = 600) -> None:
    import requests

    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited before becoming healthy (code {process.returncode})."
            )
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for vLLM health at {url}")


def stop_vllm_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()


# ---------------------------------------------------------------------------
# lm-eval model_args
# ---------------------------------------------------------------------------


def direct_vllm_model_args(task: str, model_path: str, chat_template: Path) -> dict[str, Any]:
    return {
        "pretrained": model_path,
        "trust_remote_code": True,
        "dtype": "auto",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": DIRECT_GPU_MEMORY[task],
        "max_model_len": 32768,
        "generation_config": "vllm",
        "chat_template": str(chat_template),
    }


def server_model_args(task: str, model_path: str, port: int, num_concurrent: int) -> dict[str, Any]:
    return {
        "model": model_path,
        "base_url": f"http://127.0.0.1:{port}/v1/chat/completions",
        "tokenizer_backend": "none",
        "tokenized_requests": False,
        "num_concurrent": num_concurrent,
        "max_retries": 3,
        "timeout": default_timeout(task),
        "auth_token": os.environ.get("VLLM_API_KEY", "inspectai"),
    }


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------


def extract_score(task_results: dict[str, Any], task: str) -> tuple[float | None, str | None]:
    """Return (value_0_1, metric_key) for the task's headline metric."""
    metric = SCORE_METRIC.get(task)
    if metric is None:
        return None, None
    # lm-eval metric keys carry a ``,<filter>`` suffix (e.g. ``accuracy,none``,
    # ``pass@1,create_test``, ``exact_match,k32``). Match by prefix.
    for key, value in task_results.items():
        if not isinstance(value, (int, float)):
            continue
        if key == metric or key.split(",", 1)[0] == metric:
            return float(value), key
    return None, None


# ---------------------------------------------------------------------------
# Prediction extraction for judge benches
# ---------------------------------------------------------------------------


def _sample_generation(record: dict[str, Any]) -> str:
    for key in ("filtered_resps", "resps"):
        resp = record.get(key)
        if not resp:
            continue
        item = resp[0]
        while isinstance(item, (list, tuple)) and item:
            item = item[0]
        if isinstance(item, str):
            return item
        return str(item)
    return ""


def write_predictions(task: str, samples: list[dict[str, Any]], output_dir: Path) -> int:
    predictions_path = output_dir / "predictions.jsonl"
    count = 0
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in samples:
            doc = record.get("doc") or {}
            answer = _sample_generation(record)
            if task == "healthbench":
                payload = {
                    "prompt_id": doc.get("prompt_id"),
                    "prompt": doc.get("prompt"),
                    "rubrics": doc.get("rubrics"),
                    "answer": answer,
                }
            else:  # arenahardwriting
                payload = {
                    "uid": doc.get("uid"),
                    "category": doc.get("category"),
                    "prompt": doc.get("prompt"),
                    "answer": answer,
                }
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    task = args.task
    alias = TASK_ALIASES[task]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    include_path = str(Path(args.include_path).resolve())
    templates_dir = Path(args.templates_dir).resolve()
    limit = None if args.limit < 0 else args.limit
    is_judge = task in JUDGE_TASKS

    configure_env(task, output_dir)
    chat_template = resolve_chat_template(task, args.model_name, args.model_path, templates_dir)

    # The OpenAI-compatible model must be importable/registered before
    # simple_evaluate resolves the model string.
    if task in SERVER_TASKS:
        register_tool_chat_model()

    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=include_path)

    start = time.time()
    server: subprocess.Popen | None = None
    try:
        if task in DIRECT_VLLM_TASKS:
            model = "vllm"
            model_args = direct_vllm_model_args(task, args.model_path, chat_template)
        else:
            server, port = start_vllm_server(
                task, args.model_path, chat_template, args.model_name, output_dir
            )
            model = "dataos-local-tool-chat-completions"
            model_args = server_model_args(
                task, args.model_path, port, args.num_concurrent or DEFAULT_NUM_CONCURRENT
            )

        raw = simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=[alias],
            task_manager=task_manager,
            num_fewshot=0,
            limit=limit,
            apply_chat_template=True,
            fewshot_as_multiturn=False,
            gen_kwargs=generation_kwargs(task),
            log_samples=True,
            # humaneval executes generated code; the others do not need it.
            confirm_run_unsafe_code=True,
            # Judge benches generate only here (no internet -> no judge).
            predict_only=is_judge,
        )
    finally:
        if server is not None:
            stop_vllm_server(server)

    elapsed = time.time() - start
    raw = raw if isinstance(raw, dict) else dict(raw or {})
    results = dict(raw.get("results", {}))
    samples = dict(raw.get("samples", {}))

    # Persist the raw lm-eval results payload (without the bulky samples).
    raw_to_dump = {key: value for key, value in raw.items() if key != "samples"}
    (output_dir / "lm_eval_results.json").write_text(
        json.dumps(raw_to_dump, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    task_results = results.get(alias, {})

    if is_judge:
        records = write_predictions(task, samples.get(alias, []), output_dir)
        summary = {
            "task": task,
            "model_name": args.model_name,
            "score": None,
            "needs_judge": True,
            "metrics": {"predictions": records},
            "elapsed_seconds": elapsed,
            "limit": args.limit,
        }
    else:
        value, metric_key = extract_score(task_results, task)
        score = None if value is None else value * 100.0
        summary = {
            "task": task,
            "model_name": args.model_name,
            "score": score,
            "needs_judge": False,
            "metrics": {
                key: val
                for key, val in task_results.items()
                if isinstance(val, (int, float))
            },
            "elapsed_seconds": elapsed,
            "limit": args.limit,
        }
        if metric_key is not None:
            summary["metric_key"] = metric_key

    summary_text = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    (output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
