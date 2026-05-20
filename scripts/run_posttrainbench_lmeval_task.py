from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataos.core.artifact import ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.eval.runner import run_eval_stage


TASK_ALIASES = {
    "aime2025": "posttrainbench_aime2025",
    "arenahardwriting": "posttrainbench_arena_hard_writing",
    "bfcl": "posttrainbench_bfcl",
    "gpqamain": "posttrainbench_gpqa_main",
    "gsm8k": "posttrainbench_gsm8k",
    "healthbench": "posttrainbench_healthbench",
    "humaneval": "posttrainbench_humaneval",
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_env(args, output_dir)
    judge_policy = _judge_policy(args.task, output_dir)
    if judge_policy is not None:
        (output_dir / "judge_policy.json").write_text(
            json.dumps(judge_policy, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    server = _start_vllm_server(args, output_dir)
    try:
        sandbox = Sandbox(output_dir / "dataos_artifact")
        sandbox.prepare()
        model_source = _model_source(args.model_path)
        model_artifact = ModelArtifact(
            artifact_id=args.model_name,
            name=args.model_name,
            path=Path(model_source),
            meta={"base_model": model_source, "finetuning_type": "full"},
            producer="posttrainbench_lmeval_task",
        )
        artifacts = run_eval_stage(
            [model_artifact],
            sandbox,
            task=_Task(args, model_source, f"http://127.0.0.1:{server.port}/v1/chat/completions"),
        )
        summary = dict(artifacts[0].meta)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "accuracy": summary.get("score"),
                    "results": summary.get("results", {}),
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "artifact": str(artifacts[0].path),
                    "metrics_path": str(metrics_path),
                    "score": summary.get("score"),
                    "results": summary.get("results"),
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
    finally:
        server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PostTrainBench task through lm-eval.")
    parser.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="posttrainbench-model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-concurrent", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=None)
    parser.add_argument("--posttrainbench-eval-src", default="/home/hlv/posttrainbench_eval_src")
    parser.add_argument("--benchmarks-dir", default="/home/hlv/posttrainbench_benchmarks")
    parser.add_argument("--chat-template", default="auto")
    return parser.parse_args()


class _Task:
    def __init__(self, args: argparse.Namespace, model_source: str, base_url: str) -> None:
        task_name = TASK_ALIASES[args.task]
        self.eval_config: dict[str, Any] = {
            "mode": "lm_eval_native",
            "model": "dataos-local-tool-chat-completions",
            "model_args": {
                "model": model_source,
                "base_url": base_url,
                "tokenizer_backend": "none",
                "tokenized_requests": False,
                "num_concurrent": args.num_concurrent or _default_num_concurrent(args.task),
                "max_retries": 3,
                "timeout": _default_timeout(args.task),
                "auth_token": os.environ.get("VLLM_API_KEY", "inspectai"),
            },
            "tasks": [task_name],
            "include_path": str(PROJECT_ROOT / "dataos" / "eval_tasks" / "posttrainbench"),
            "apply_chat_template": True,
            "fewshot_as_multiturn": False,
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_tokens,
            "limit": None if args.limit < 0 else args.limit,
            "gen_kwargs": _generation_kwargs(args.task, args.max_tokens),
            "log_samples": True,
            "confirm_run_unsafe_code": args.task == "humaneval",
            "metadata": {
                "benchmark": "PostTrainBench",
                "task": args.task,
                "task_name": task_name,
                "execution": "lm_eval_native_local_chat_completions",
                "max_gen_toks": args.max_tokens,
                "chat_template": str(_resolve_chat_template(args)),
                "vllm_generation_config": "vllm",
                "task_generation_kwargs": _task_generation_kwargs(args.task, args.max_tokens),
                "judge_policy": _judge_policy(args.task, Path(args.output_dir).resolve()),
                "gpqa_frozen_jsonl": os.environ.get("POSTTRAINBENCH_GPQA_FROZEN_JSONL")
                if args.task == "gpqamain"
                else None,
                "gpqa_shuffle_seed": os.environ.get("POSTTRAINBENCH_GPQA_SHUFFLE_SEED")
                if args.task == "gpqamain"
                else None,
            },
        }


class _Server:
    def __init__(self, process: subprocess.Popen[str], port: int) -> None:
        self.process = process
        self.port = port

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()


def _configure_env(args: argparse.Namespace, output_dir: Path) -> None:
    os.environ.setdefault("VLLM_API_KEY", "inspectai")
    os.environ["POSTTRAINBENCH_EVAL_SRC"] = args.posttrainbench_eval_src
    os.environ["POSTTRAINBENCH_BENCHMARKS_DIR"] = args.benchmarks_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(args.benchmarks_dir) / "hf_datasets"))
    if args.task == "humaneval":
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    if args.task == "gpqamain":
        frozen = Path(args.benchmarks_dir) / "local" / "gpqa_main_choice_order_20260512.jsonl"
        if frozen.exists():
            os.environ.setdefault("POSTTRAINBENCH_GPQA_FROZEN_JSONL", str(frozen))
        os.environ.setdefault("POSTTRAINBENCH_GPQA_SHUFFLE_SEED", "42")
    if args.task == "aime2025":
        os.environ.setdefault("DATAOS_LMEVAL_API_WINDOW", "32")
        os.environ.setdefault(
            "POSTTRAINBENCH_AIME_EXTRACTION_LOG",
            str(output_dir / "aime_extraction_events.jsonl"),
        )
    if args.task in {"arenahardwriting", "healthbench"}:
        _configure_openai_judge_env()
    if args.task == "arenahardwriting":
        os.environ.setdefault(
            "POSTTRAINBENCH_ARENA_JUDGE_LOG",
            str(output_dir / "arena_judge_events.jsonl"),
        )
        os.environ.setdefault("POSTTRAINBENCH_ARENA_JUDGE_MAX_RETRIES", "5")
        os.environ.setdefault("POSTTRAINBENCH_ARENA_JUDGE_RETRY_SLEEP", "2")
        os.environ.setdefault("POSTTRAINBENCH_ARENA_JUDGE_MAX_TOKENS", "16000")
        os.environ.setdefault("POSTTRAINBENCH_ARENA_JUDGE_CONCURRENCY", "1")
    if args.task == "healthbench":
        os.environ.setdefault(
            "POSTTRAINBENCH_HEALTHBENCH_JUDGE_LOG",
            str(output_dir / "healthbench_judge_events.jsonl"),
        )
        os.environ.setdefault("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_RETRIES", "5")
        os.environ.setdefault("POSTTRAINBENCH_HEALTHBENCH_JUDGE_RETRY_SLEEP", "2")
        os.environ.setdefault("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_TOKENS", "512")
        os.environ.setdefault("POSTTRAINBENCH_HEALTHBENCH_JUDGE_CONCURRENCY", "4")
        os.environ.setdefault("POSTTRAINBENCH_HEALTHBENCH_CRITERIA_WORKERS", "8")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _configure_openai_judge_env() -> None:
    judge_model = os.environ.get("POSTTRAINBENCH_JUDGE_MODEL") or os.environ.get(
        "BAILIAN_JUDGE_MODEL"
    )
    if not judge_model:
        judge_model = "qwen3.5-plus"
    if judge_model:
        os.environ.setdefault("POSTTRAINBENCH_JUDGE_MODEL", judge_model)

    judge_api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("POSTTRAINBENCH_JUDGE_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("BAILIAN_API_KEY")
    )
    if judge_api_key:
        os.environ.setdefault("OPENAI_API_KEY", judge_api_key)

    judge_base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("POSTTRAINBENCH_JUDGE_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
    )
    if judge_base_url:
        os.environ.setdefault("OPENAI_BASE_URL", judge_base_url)
    elif os.environ.get("POSTTRAINBENCH_JUDGE_MODEL", "").lower().startswith("qwen"):
        os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def _judge_policy(task: str, output_dir: Path) -> dict[str, Any] | None:
    if task not in {"arenahardwriting", "healthbench"}:
        return None
    policy: dict[str, Any] = {
        "model": os.environ.get("POSTTRAINBENCH_JUDGE_MODEL", "gpt-5-mini"),
        "base_url": os.environ.get("OPENAI_BASE_URL") or "openai_default",
        "api_key_env": "OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY") else None,
    }
    if task == "arenahardwriting":
        policy.update(
            {
                "official_judge_model": "gpt-5-mini",
                "dataos_judge_model": os.environ.get("POSTTRAINBENCH_JUDGE_MODEL", "gpt-5-mini"),
                "official_pairwise_prompt": "arena-hard-v2.0.yaml prompt_template plus category system prompt",
                "baseline_policy": "Qwen3-1.7B for creative_writing; o3-mini-2025-01-31 for hard_prompt/coding/math",
                "judge_games_per_sample": 2,
                "max_retries": int(os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_MAX_RETRIES", "5")),
                "retry_sleep_seconds": float(
                    os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_RETRY_SLEEP", "2")
                ),
                "temperature_for_non_gpt_judge": 0,
                "max_tokens_for_non_gpt_judge": int(
                    os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_MAX_TOKENS", "16000")
                ),
                "max_completion_tokens_for_gpt_o_series": int(
                    os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_GPT_MAX_COMPLETION_TOKENS", "49152")
                ),
                "max_concurrent_requests": int(
                    os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_CONCURRENCY", "1")
                ),
                "failure_policy": "retry judge requests; abort on 401/429/billing/quota/rate-limit or exhausted retries",
                "evidence_log": str(output_dir / "arena_judge_events.jsonl"),
            }
        )
        return policy
    policy.update(
        {
            "max_retries": int(os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_RETRIES", "5")),
            "retry_sleep_seconds": float(
                os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_RETRY_SLEEP", "2")
            ),
            "max_tokens_for_non_gpt_judge": int(
                os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_TOKENS", "512")
            ),
            "max_completion_tokens_for_gpt_o_series": int(
                os.environ.get("POSTTRAINBENCH_HEALTHBENCH_GPT_MAX_COMPLETION_TOKENS", "49152")
            ),
            "temperature_for_non_gpt_judge": 0,
            "criteria_workers": int(
                os.environ.get("POSTTRAINBENCH_HEALTHBENCH_CRITERIA_WORKERS", "8")
            ),
            "max_concurrent_requests": int(
                os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_CONCURRENCY", "4")
            ),
            "failure_policy": "retry judge requests; abort on 401/429/billing/quota/rate-limit or exhausted retries",
            "evidence_log": str(output_dir / "healthbench_judge_events.jsonl"),
        }
    )
    return policy


def _start_vllm_server(args: argparse.Namespace, output_dir: Path) -> _Server:
    port = args.vllm_port or _find_free_port()
    vllm_bin = Path(sys.executable).with_name("vllm")
    command = [
        str(vllm_bin) if vllm_bin.exists() else "vllm",
        "serve",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--api-key",
        os.environ.get("VLLM_API_KEY", "inspectai"),
        "--chat-template",
        str(_resolve_chat_template(args)),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--generation-config",
        "vllm",
    ]
    if args.task == "bfcl":
        command.extend(
            [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                _tool_call_parser_name(args.model_path),
            ]
        )
    log_path = output_dir / "vllm_server.log"
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for_health(port, process)
    except Exception:
        process.kill()
        raise
    return _Server(process, port)


def _wait_for_health(port: int, process: subprocess.Popen[str]) -> None:
    import requests

    deadline = time.time() + 600
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM server exited before becoming healthy.")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("Timed out waiting for vLLM health endpoint.")


def _find_free_port() -> int:
    for _ in range(100):
        port = random.randint(20000, 65000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Could not find a free local port.")


def _resolve_chat_template(args: argparse.Namespace) -> Path:
    if args.chat_template != "auto":
        return Path(args.chat_template)
    template = _template_name(args.model_path, args.task)
    return PROJECT_ROOT / "dataos" / "eval_tasks" / "posttrainbench" / "templates" / template


def _template_name(model_path: str, task: str) -> str:
    lower = model_path.lower()
    if "qwen" in lower:
        return "qwen3.jinja"
    if "smollm" in lower:
        return "smollm.jinja"
    if "gemma" in lower:
        return "gemma3_tool_calling.jinja" if task == "bfcl" else "gemma3.jinja"
    raise ValueError(f"Cannot infer chat template for model path: {model_path}")


def _tool_call_parser_name(model_path: str) -> str:
    lower = model_path.lower()
    if any(name in lower for name in ("qwen", "gemma", "smollm")):
        return "hermes"
    if "llama" in lower:
        return "llama3_json"
    return "hermes"


def _default_num_concurrent(task: str) -> int:
    if task in {"arenahardwriting", "bfcl", "healthbench", "humaneval"}:
        return 1
    if task == "aime2025":
        return 2
    return 6


def _default_timeout(task: str) -> int:
    if task == "aime2025":
        return 18_000_000
    if task in {"arenahardwriting", "bfcl", "gpqamain", "healthbench"}:
        return 1800
    if task == "humaneval":
        return 900
    return 300


def _model_source(model_path: str) -> str:
    path = Path(model_path).expanduser()
    return str(path.resolve()) if path.exists() else model_path


def _generation_kwargs(task: str, max_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_gen_toks": max_tokens}
    if task == "bfcl":
        kwargs.update(
            {
                "temperature": 0.6,
                "top_p": 0.95,
                "until": ["<|im_end|>", "<|eot_id|>"],
            }
        )
    if task == "aime2025":
        kwargs["temperature"] = 0.6
        kwargs["top_p"] = 0.95
        kwargs["dataos_omit_seed"] = True
    if task == "gpqamain":
        kwargs["temperature"] = 1.0
    if task in {"arenahardwriting", "healthbench"}:
        kwargs["dataos_official_chat_payload"] = True
    return kwargs


def _task_generation_kwargs(task: str, max_tokens: int) -> dict[str, Any]:
    return _generation_kwargs(task, max_tokens)


if __name__ == "__main__":
    main()
