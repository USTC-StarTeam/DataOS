from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    runner: str
    default_max_tokens: int
    default_max_connections: int
    default_gpu_memory: float
    default_max_model_len: int
    requires_api_key: bool = False


TASKS: dict[str, TaskSpec] = {
    "aime2025": TaskSpec("aime2025", "dataos_lmeval_native", 16000, 6, 0.8, 32768),
    "arenahardwriting": TaskSpec(
        "arenahardwriting", "dataos_lmeval_native", 16384, 1, 0.8, 32768, True
    ),
    "bfcl": TaskSpec("bfcl", "dataos_lmeval_native", 16000, 1, 0.8, 32768),
    "gpqamain": TaskSpec("gpqamain", "direct_vllm_lmeval", 16000, 6, 0.8, 32768),
    "gsm8k": TaskSpec("gsm8k", "gsm8k_vllm_native", 4000, 2, 0.5, 32768),
    "healthbench": TaskSpec("healthbench", "dataos_lmeval_native", 16384, 1, 0.8, 32768, True),
    "humaneval": TaskSpec("humaneval", "direct_vllm_lmeval", 4000, 1, 0.3, 8192),
}

INSPECT_SERVER_RUNNERS = {"aime_k32", "posttrainbench_script"}


def main() -> None:
    args = parse_args()
    spec = _resolve_spec(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    protocol_path = output_dir / "protocol.json"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"

    env = _build_env(args)
    if args.task in {"arenahardwriting", "healthbench"}:
        env.setdefault("POSTTRAINBENCH_JUDGE_MODEL", "qwen3.5-plus")
        env.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if args.task == "arenahardwriting":
        env.setdefault(
            "POSTTRAINBENCH_ARENA_JUDGE_LOG",
            str(output_dir / "arena_judge_events.jsonl"),
        )
        env.setdefault("POSTTRAINBENCH_ARENA_JUDGE_MAX_RETRIES", "5")
        env.setdefault("POSTTRAINBENCH_ARENA_JUDGE_RETRY_SLEEP", "2")
        env.setdefault("POSTTRAINBENCH_ARENA_JUDGE_MAX_TOKENS", "16000")
        env.setdefault("POSTTRAINBENCH_ARENA_JUDGE_CONCURRENCY", "1")
    if args.task == "healthbench":
        env.setdefault(
            "POSTTRAINBENCH_HEALTHBENCH_JUDGE_LOG",
            str(output_dir / "healthbench_judge_events.jsonl"),
        )
        env.setdefault("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_RETRIES", "5")
        env.setdefault("POSTTRAINBENCH_HEALTHBENCH_JUDGE_RETRY_SLEEP", "2")
        env.setdefault("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_TOKENS", "512")
        env.setdefault("POSTTRAINBENCH_HEALTHBENCH_JUDGE_CONCURRENCY", "4")
        env.setdefault("POSTTRAINBENCH_HEALTHBENCH_CRITERIA_WORKERS", "8")
    command = _build_command(args, spec, metrics_path, output_dir)
    protocol = {
        "benchmark": "PostTrainBench",
        "task": args.task,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "runner": spec.runner,
        "runner_override": args.runner_override,
        "limit": args.limit,
        "epochs": args.epochs if args.task == "aime2025" else None,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_connections": args.max_connections,
        "vllm_generation_config": "vllm",
        "task_generation_kwargs": _task_generation_kwargs(args.task, args.max_tokens),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "posttrainbench_eval_src": args.posttrainbench_eval_src,
        "benchmarks_dir": args.benchmarks_dir,
        "inspect_pkgs": args.inspect_pkgs,
        "gpqa_shuffle_seed": env.get("POSTTRAINBENCH_GPQA_SHUFFLE_SEED")
        if args.task == "gpqamain"
        else None,
        "gpqa_frozen_jsonl": env.get("POSTTRAINBENCH_GPQA_FROZEN_JSONL")
        if args.task == "gpqamain"
        else None,
        "judge_policy": _judge_policy(args.task, env, output_dir),
        "aime_policy": _aime_policy(args) if args.task == "aime2025" else None,
        "requires_api_key": spec.requires_api_key,
        "command": command,
        "env_keys": sorted(
            k
            for k in env
            if k.startswith(("DASHSCOPE", "HF_", "OPENAI", "POSTTRAIN", "VLLM_", "PYTHON"))
        ),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    if spec.requires_api_key and not _has_judge_credentials(env):
        raise SystemExit(
            f"{args.task} requires OPENAI-compatible judge credentials for its judge path. "
            f"Protocol was written to {protocol_path}."
        )

    server: subprocess.Popen[str] | None = None
    start = time.time()
    try:
        if spec.runner in INSPECT_SERVER_RUNNERS and args.task not in {
            "arenahardwriting",
            "healthbench",
        }:
            server, base_url = _start_vllm_server(args, spec, env, output_dir)
            env["VLLM_BASE_URL"] = base_url
        os.environ.update(env)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            proc = subprocess.run(
                command,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
    elapsed = time.time() - start

    summary: dict[str, Any] = {
        "task": args.task,
        "model_name": args.model_name,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "metrics_path": str(metrics_path),
        "protocol_path": str(protocol_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if not metrics_path.exists() and spec.runner in {"dataos_lmeval_native", "gsm8k_vllm_native"}:
        _write_metrics_from_dataos_artifact(output_dir, metrics_path)
    if metrics_path.exists():
        summary["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        accuracy = summary["metrics"].get("accuracy")
        if isinstance(accuracy, (int, float)):
            summary["score"] = accuracy if accuracy > 1.0 else accuracy * 100.0
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _write_metrics_from_dataos_artifact(output_dir: Path, metrics_path: Path) -> None:
    eval_dirs = sorted((output_dir / "dataos_artifact" / "artifacts" / "evals").glob("eval-*"))
    if not eval_dirs:
        return
    result_path = eval_dirs[-1] / "results.json"
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    score = result.get("score")
    metrics: dict[str, Any] = {
        "accuracy": score,
        "source_results_json": str(result_path),
        "source_protocol_json": str(eval_dirs[-1] / "protocol.json"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PostTrainBench protocol on 5090.")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--posttrainbench-eval-src", default="/home/hlv/posttrainbench_eval_src")
    parser.add_argument("--benchmarks-dir", default="/home/hlv/posttrainbench_benchmarks")
    parser.add_argument("--inspect-pkgs", default="/home/hlv/inspect_pkgs")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-connections", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=None)
    parser.add_argument(
        "--runner-override",
        choices=[
            "aime_k32",
            "dataos_lmeval_native",
            "direct_vllm_lmeval",
            "gsm8k_vllm_native",
            "posttrainbench_script",
        ],
        default=None,
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    spec = _resolve_spec(args)
    args.max_tokens = args.max_tokens or spec.default_max_tokens
    args.max_model_len = args.max_model_len or spec.default_max_model_len
    args.max_connections = args.max_connections or spec.default_max_connections
    args.gpu_memory_utilization = args.gpu_memory_utilization or _default_gpu_memory(args, spec)
    return args


def _resolve_spec(args: argparse.Namespace) -> TaskSpec:
    spec = TASKS[args.task]
    if args.runner_override:
        return replace(spec, runner=args.runner_override)
    return spec


def _default_gpu_memory(args: argparse.Namespace, spec: TaskSpec) -> float:
    if args.task == "gsm8k":
        lowered = f"{args.model_name} {args.model_path}".lower()
        if "qwen3-1.7b" in lowered:
            return 0.3
        return 0.5
    return spec.default_gpu_memory


def _task_generation_kwargs(task: str, max_tokens: int) -> dict[str, Any]:
    if task == "bfcl":
        return {
            "max_gen_toks": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "until": ["<|im_end|>", "<|eot_id|>"],
        }
    if task == "aime2025":
        return {
            "max_gen_toks": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "dataos_omit_seed": True,
        }
    if task == "gpqamain":
        return {"max_gen_toks": max_tokens, "temperature": 1.0}
    if task in {"arenahardwriting", "healthbench"}:
        return {"max_gen_toks": max_tokens, "dataos_official_chat_payload": True}
    return {"max_gen_toks": max_tokens}


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "HOME": env.get("HOME", "/home/hlv"),
            "HF_HOME": env.get("HF_HOME", "/home/hlv/dataos_eval_hf_cache"),
            "HF_ENDPOINT": env.get("HF_ENDPOINT", "https://hf-mirror.com"),
            "POSTTRAINBENCH_BENCHMARKS_DIR": args.benchmarks_dir,
            "HF_DATASETS_CACHE": str(Path(args.benchmarks_dir) / "hf_datasets"),
            "VLLM_API_KEY": env.get("VLLM_API_KEY", "inspectai"),
            "VLLM_WORKER_MULTIPROC_METHOD": env.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    pythonpath = [args.inspect_pkgs, env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = ":".join(part for part in pythonpath if part)
    bin_dir = str(Path(args.python).parent)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    if args.task == "gpqamain":
        frozen = Path(args.benchmarks_dir) / "local" / "gpqa_main_choice_order_20260512.jsonl"
        if frozen.exists():
            env.setdefault("POSTTRAINBENCH_GPQA_FROZEN_JSONL", str(frozen))
        env.setdefault("POSTTRAINBENCH_GPQA_SHUFFLE_SEED", "42")
    if args.task == "aime2025":
        env.setdefault(
            "POSTTRAINBENCH_AIME_EXTRACTION_LOG",
            str(Path(args.output_dir).resolve() / "aime_extraction_events.jsonl"),
        )
    _configure_judge_env(env)
    return env


def _aime_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol_name": "dataos_fixed_k32",
        "dataset_path": "math-ai/aime25",
        "split": "test",
        "num_questions": 30,
        "repeats": 32,
        "max_tokens": args.max_tokens,
        "max_connections": args.max_connections,
        "temperature": 0.6,
        "top_p": 0.95,
        "prompt_template": "inspect_evals.aime2025 USER_PROMPT_TEMPLATE",
        "seed_policy": "omit OpenAI-compatible seed so K32 repeats are independent samples",
        "scorer": "numeric match after last boxed answer or final ANSWER line",
        "evidence_log": str(Path(args.output_dir).resolve() / "aime_extraction_events.jsonl"),
        "note": "K32 is the DataOS fixed protocol for PostTrainBench AIME; do not mix it with single-sample scores.",
    }


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


def _judge_policy(task: str, env: dict[str, str], output_dir: Path) -> dict[str, Any] | None:
    if task not in {"healthbench", "arenahardwriting"}:
        return None
    policy: dict[str, Any] = {
        "model": env.get("POSTTRAINBENCH_JUDGE_MODEL") or "gpt-5-mini",
        "base_url": env.get("OPENAI_BASE_URL") or "openai_default",
        "api_key_env": "OPENAI_API_KEY" if env.get("OPENAI_API_KEY") else None,
    }
    if task == "arenahardwriting":
        policy.update(
            {
                "official_judge_model": "gpt-5-mini",
                "dataos_judge_model": env.get("POSTTRAINBENCH_JUDGE_MODEL") or "gpt-5-mini",
                "official_pairwise_prompt": "arena-hard-v2.0.yaml prompt_template plus category system prompt",
                "baseline_policy": "Qwen3-1.7B for creative_writing; o3-mini-2025-01-31 for hard_prompt/coding/math",
                "judge_games_per_sample": 2,
                "official_judge_workers": 64,
                "dataos_judge_concurrency": int(
                    env.get("POSTTRAINBENCH_ARENA_JUDGE_CONCURRENCY", "1")
                ),
                "max_retries": int(env.get("POSTTRAINBENCH_ARENA_JUDGE_MAX_RETRIES", "5")),
                "retry_sleep_seconds": float(
                    env.get("POSTTRAINBENCH_ARENA_JUDGE_RETRY_SLEEP", "2")
                ),
                "temperature_for_non_gpt_judge": 0,
                "official_config_max_tokens": 16000,
                "max_tokens_for_non_gpt_judge": int(
                    env.get("POSTTRAINBENCH_ARENA_JUDGE_MAX_TOKENS", "16000")
                ),
                "max_completion_tokens_for_gpt_o_series": int(
                    env.get("POSTTRAINBENCH_ARENA_JUDGE_GPT_MAX_COMPLETION_TOKENS", "49152")
                ),
                "failure_policy": (
                    "retry judge requests; abort on 401/429/billing/quota/rate-limit or "
                    "exhausted retries"
                ),
                "evidence_log": str(output_dir / "arena_judge_events.jsonl"),
                "strict_reproduction_note": (
                    "Official PostTrainBench uses gpt-5-mini; DataOS uses the configured "
                    "OpenAI-compatible judge for this project and records the difference."
                ),
            }
        )
        return policy
    if task == "healthbench":
        policy.update(
            {
                "official_judge_model": "gpt-5-mini",
                "dataos_judge_model": env.get("POSTTRAINBENCH_JUDGE_MODEL") or "gpt-5-mini",
                "max_retries": int(
                    env.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_RETRIES", "5")
                ),
                "retry_sleep_seconds": float(
                    env.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_RETRY_SLEEP", "2")
                ),
                "temperature_for_non_gpt_judge": 0,
                "max_tokens_for_non_gpt_judge": int(
                    env.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_TOKENS", "512")
                ),
                "max_completion_tokens_for_gpt_o_series": int(
                    env.get(
                        "POSTTRAINBENCH_HEALTHBENCH_GPT_MAX_COMPLETION_TOKENS",
                        "49152",
                    )
                ),
                "criteria_workers": int(
                    env.get("POSTTRAINBENCH_HEALTHBENCH_CRITERIA_WORKERS", "8")
                ),
                "max_concurrent_requests": int(
                    env.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_CONCURRENCY", "4")
                ),
                "failure_policy": (
                    "retry judge requests; abort on 401/429/billing/quota/rate-limit or "
                    "exhausted retries"
                ),
                "evidence_log": str(output_dir / "healthbench_judge_events.jsonl"),
            }
        )
    return policy


def _start_vllm_server(
    args: argparse.Namespace,
    spec: TaskSpec,
    env: dict[str, str],
    output_dir: Path,
) -> tuple[subprocess.Popen[str], str]:
    del spec
    port = args.vllm_port or _find_free_port()
    template = Path(args.posttrainbench_eval_src) / "templates" / _template_name(
        args.model_path,
        args.task,
    )
    vllm_bin = Path(args.python).parent / "vllm"
    command = [
        str(vllm_bin),
        "serve",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--api-key",
        env["VLLM_API_KEY"],
        "--chat-template",
        str(template),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
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
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for_health(port, proc)
    except Exception:
        if proc.poll() is None:
            proc.kill()
        raise
    return proc, f"http://127.0.0.1:{port}/v1"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, proc: subprocess.Popen[str], timeout: int = 600) -> None:
    import requests

    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for vLLM health at {url}")


def _model_family(model_path: str) -> str:
    lowered = model_path.lower()
    if "qwen" in lowered:
        return "qwen"
    if "smollm" in lowered:
        return "smollm"
    if "gemma" in lowered:
        return "gemma"
    if "llama" in lowered:
        return "llama"
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        architecture = json.loads(config_path.read_text(encoding="utf-8")).get(
            "architectures",
            [""],
        )[0].lower()
        if "qwen" in architecture:
            return "qwen"
        if "smollm" in architecture:
            return "smollm"
        if "gemma" in architecture:
            return "gemma"
        if "llama" in architecture:
            return "llama"
    raise ValueError(f"Cannot infer model family from {model_path}")


def _template_name(model_path: str, task: str) -> str:
    family = _model_family(model_path)
    if family == "qwen":
        return "qwen3.jinja"
    if family == "smollm":
        return "smollm.jinja"
    if family == "gemma":
        return "gemma3_tool_calling.jinja" if task == "bfcl" else "gemma3.jinja"
    if family == "llama":
        return "llama3.jinja"
    raise ValueError(family)


def _tool_call_parser_name(model_path: str) -> str:
    family = _model_family(model_path)
    if family in {"gemma", "qwen", "smollm"}:
        return "hermes"
    if family == "llama":
        return "llama3_json"
    raise ValueError(family)


def _build_command(
    args: argparse.Namespace,
    spec: TaskSpec,
    metrics_path: Path,
    output_dir: Path,
) -> list[str]:
    eval_src = Path(args.posttrainbench_eval_src)
    templates_dir = eval_src / "templates"
    if spec.runner == "direct_vllm_lmeval":
        return [
            args.python,
            str(Path(__file__).with_name("run_posttrainbench_direct_lmeval_task.py")),
            "--task",
            args.task,
            "--model-path",
            args.model_path,
            "--model-name",
            args.model_name,
            "--output-dir",
            str(output_dir),
            "--limit",
            str(args.limit),
            "--batch-size",
            args.batch_size,
            "--max-tokens",
            str(args.max_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--chat-template",
            "auto",
            "--json-output-file",
            str(metrics_path),
        ]
    if spec.runner == "dataos_lmeval_native":
        return [
            args.python,
            str(Path(__file__).with_name("run_posttrainbench_lmeval_task.py")),
            "--task",
            args.task,
            "--model-path",
            args.model_path,
            "--model-name",
            args.model_name,
            "--output-dir",
            str(output_dir),
            "--limit",
            str(args.limit),
            "--batch-size",
            args.batch_size,
            "--max-tokens",
            str(args.max_tokens),
            "--num-concurrent",
            str(args.max_connections),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--posttrainbench-eval-src",
            args.posttrainbench_eval_src,
            "--benchmarks-dir",
            args.benchmarks_dir,
        ]
    if spec.runner == "gsm8k_vllm_native":
        return [
            args.python,
            str(Path(__file__).with_name("run_posttrainbench_gsm8k_eval.py")),
            "--model-path",
            args.model_path,
            "--model-name",
            args.model_name,
            "--output-dir",
            str(output_dir),
            "--backend",
            "vllm",
            "--limit",
            str(args.limit),
            "--batch-size",
            args.batch_size,
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-gen-toks",
            str(args.max_tokens),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--json-output-file",
            str(metrics_path),
        ]
    if spec.runner == "aime_k32":
        return [
            args.python,
            str(Path(__file__).with_name("run_posttrainbench_aime_k32_eval.py")),
            "--model-path",
            args.model_path,
            "--templates-dir",
            str(templates_dir),
            "--limit",
            str(args.limit),
            "--epochs",
            str(args.epochs),
            "--max-tokens",
            str(args.max_tokens),
            "--max-connections",
            str(args.max_connections),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--log-dir",
            str(output_dir / "logs"),
            "--json-output-file",
            str(metrics_path),
        ]

    evaluate_py = eval_src / "tasks" / args.task / "evaluate.py"
    command = [
        args.python,
        str(evaluate_py),
        "--model-path",
        args.model_path,
        "--templates-dir",
        str(templates_dir),
        "--limit",
        str(args.limit),
        "--json-output-file",
        str(metrics_path),
    ]
    if args.task in {"arenahardwriting", "healthbench"}:
        command.extend(["--max-new-tokens", str(args.max_tokens)])
    else:
        command.extend(
            [
                "--max-tokens",
                str(args.max_tokens),
                "--max-connections",
                str(args.max_connections),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
        )
    if args.task in {"arenahardwriting", "healthbench"}:
        command.extend(["--store-outputs"])
    return command


if __name__ == "__main__":
    main()
