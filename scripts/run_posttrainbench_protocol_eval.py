from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    runner: str
    default_max_tokens: int
    default_max_connections: int
    default_gpu_memory: float
    requires_api_key: bool = False


TASKS: dict[str, TaskSpec] = {
    "aime2025": TaskSpec("aime2025", "aime_k32", 30000, 6, 0.8),
    "arenahardwriting": TaskSpec("arenahardwriting", "posttrainbench_script", 16384, 1, 0.8, True),
    "bfcl": TaskSpec("bfcl", "posttrainbench_script", 16000, 6, 0.8),
    "gpqamain": TaskSpec("gpqamain", "posttrainbench_script", 16000, 6, 0.8),
    "gsm8k": TaskSpec("gsm8k", "dataos_lmeval_native", 4000, 2, 0.3),
    "healthbench": TaskSpec("healthbench", "posttrainbench_script", 16384, 1, 0.8, True),
    "humaneval": TaskSpec("humaneval", "posttrainbench_script", 4000, 1, 0.3),
}

INSPECT_SERVER_RUNNERS = {"aime_k32", "posttrainbench_script"}


def main() -> None:
    args = parse_args()
    spec = TASKS[args.task]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    protocol_path = output_dir / "protocol.json"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"

    env = _build_env(args)
    command = _build_command(args, spec, metrics_path, output_dir)
    protocol = {
        "benchmark": "PostTrainBench",
        "task": args.task,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "runner": spec.runner,
        "limit": args.limit,
        "epochs": args.epochs if args.task == "aime2025" else None,
        "max_tokens": args.max_tokens,
        "max_connections": args.max_connections,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "posttrainbench_eval_src": args.posttrainbench_eval_src,
        "benchmarks_dir": args.benchmarks_dir,
        "inspect_pkgs": args.inspect_pkgs,
        "requires_api_key": spec.requires_api_key,
        "command": command,
        "env_keys": sorted(k for k in env if k.startswith(("HF_", "POSTTRAIN", "VLLM_", "PYTHON"))),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    if spec.requires_api_key and not env.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"{args.task} requires OPENAI_API_KEY for its judge path. "
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
    if not metrics_path.exists() and spec.runner == "dataos_lmeval_native":
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
    parser.add_argument("--max-connections", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    spec = TASKS[args.task]
    args.max_tokens = args.max_tokens or spec.default_max_tokens
    args.max_connections = args.max_connections or spec.default_max_connections
    args.gpu_memory_utilization = args.gpu_memory_utilization or spec.default_gpu_memory
    return args


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
    return env


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
    if spec.runner == "dataos_lmeval_native":
        return [
            args.python,
            str(Path(__file__).with_name("run_posttrainbench_gsm8k_eval.py")),
            "--model-path",
            args.model_path,
            "--model-name",
            args.model_name,
            "--output-dir",
            str(output_dir / "dataos_artifact"),
            "--backend",
            "vllm",
            "--limit",
            str(args.limit),
            "--batch-size",
            args.batch_size,
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--max-gen-toks",
            str(args.max_tokens),
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
