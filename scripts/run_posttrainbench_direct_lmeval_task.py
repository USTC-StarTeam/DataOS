from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataos.core.artifact import ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.eval.runner import run_eval_stage


TASK_ALIASES = {
    "gpqamain": "posttrainbench_gpqa_main",
    "humaneval": "posttrainbench_humaneval",
}


def main() -> None:
    args = parse_args()
    _configure_env(args)
    output_dir = Path(args.output_dir).resolve()
    sandbox = Sandbox(output_dir)
    sandbox.prepare()
    model_path = Path(args.model_path).expanduser()
    model_source = str(model_path.resolve()) if model_path.exists() else args.model_path
    model_artifact = ModelArtifact(
        artifact_id=args.model_name,
        name=args.model_name,
        path=Path(model_source),
        meta={"base_model": model_source, "finetuning_type": "full"},
        producer="posttrainbench_direct_lmeval_task",
    )
    artifact = run_eval_stage([model_artifact], sandbox, task=_Task(args, model_source))
    summary = dict(artifact[0].meta)
    if args.json_output_file:
        output = {
            "accuracy": summary.get("score"),
            "results": summary.get("results", {}),
            "artifact": str(artifact[0].path),
            "source_results_json": str(artifact[0].path / "results.json"),
            "source_protocol_json": str(artifact[0].path / "protocol.json"),
            "task_protocol": f"{args.task}_direct_vllm_lm_eval_native",
        }
        Path(args.json_output_file).write_text(
            json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "artifact": str(artifact[0].path),
                "score": summary.get("score"),
                "results": summary.get("results"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DataOS lm-eval native with lm-eval's direct vLLM backend."
    )
    parser.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="posttrainbench-model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Diagnostic override for lm-eval generation temperature. Omit to use the fixed task protocol.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override lm-eval nucleus sampling top_p.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override lm-eval top_k.",
    )
    parser.add_argument(
        "--do-sample",
        choices=("auto", "true", "false"),
        default="auto",
        help="Diagnostic override for do_sample. auto uses the fixed task protocol.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=-1,
        help="Seed for the lm-eval vLLM backend. Negative means leave the backend default unset.",
    )
    parser.add_argument("--json-output-file", default=None)
    parser.add_argument("--chat-template", default="auto")
    parser.add_argument("--posttrainbench-eval-src", default="/home/hlv/posttrainbench_eval_src")
    parser.add_argument("--benchmarks-dir", default="/home/hlv/posttrainbench_benchmarks")
    return parser.parse_args()


class _Task:
    def __init__(self, args: argparse.Namespace, model_source: str) -> None:
        task_name = TASK_ALIASES[args.task]
        chat_template = _resolve_chat_template(args.chat_template, model_source, args.model_name)
        model_args: dict[str, Any] = {
            "pretrained": model_source,
            "trust_remote_code": True,
            "dtype": "auto",
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "generation_config": "vllm",
        }
        if args.model_seed >= 0:
            model_args["seed"] = args.model_seed
        if chat_template is not None:
            model_args["chat_template"] = str(chat_template)
        self.eval_config = {
            "mode": "lm_eval_native",
            "model": "vllm",
            "model_args": model_args,
            "tasks": [task_name],
            "include_path": str(PROJECT_ROOT / "dataos" / "eval_tasks" / "posttrainbench"),
            "apply_chat_template": chat_template is not None,
            "fewshot_as_multiturn": False,
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "limit": None if args.limit < 0 else args.limit,
            "gen_kwargs": _generation_kwargs(args),
            "log_samples": True,
            "confirm_run_unsafe_code": args.task == "humaneval",
            "metadata": {
                "benchmark": "PostTrainBench",
                "task": args.task,
                "task_name": task_name,
                "execution": "lm_eval_native_direct_vllm",
                "max_gen_toks": args.max_tokens,
                "vllm_generation_config": "vllm",
                "chat_template": str(chat_template) if chat_template is not None else None,
                "model_seed": args.model_seed if args.model_seed >= 0 else None,
                "generation_overrides": _generation_overrides_metadata(args),
                "gpqa_frozen_jsonl": os.environ.get("POSTTRAINBENCH_GPQA_FROZEN_JSONL")
                if args.task == "gpqamain"
                else None,
                "gpqa_shuffle_seed": os.environ.get("POSTTRAINBENCH_GPQA_SHUFFLE_SEED")
                if args.task == "gpqamain"
                else None,
            },
        }


def _configure_env(args: argparse.Namespace) -> None:
    os.environ["POSTTRAINBENCH_EVAL_SRC"] = args.posttrainbench_eval_src
    os.environ["POSTTRAINBENCH_BENCHMARKS_DIR"] = args.benchmarks_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(args.benchmarks_dir) / "hf_datasets"))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if args.task == "humaneval":
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    if args.task == "gpqamain":
        frozen = Path(args.benchmarks_dir) / "local" / "gpqa_main_choice_order_20260512.jsonl"
        if frozen.exists():
            os.environ.setdefault("POSTTRAINBENCH_GPQA_FROZEN_JSONL", str(frozen))
        os.environ.setdefault("POSTTRAINBENCH_GPQA_SHUFFLE_SEED", "42")


def _resolve_chat_template(value: str, model_source: str, model_name: str) -> Path | None:
    normalized = value.strip()
    if normalized.lower() in {"", "none", "false", "0"}:
        return None
    if normalized.lower() != "auto":
        template = Path(normalized).expanduser()
        if not template.exists():
            raise FileNotFoundError(f"Chat template does not exist: {template}")
        return template.resolve()
    template_name = _template_name_for_model(f"{model_name} {model_source}")
    template = PROJECT_ROOT / "dataos" / "eval_tasks" / "posttrainbench" / "templates" / template_name
    if not template.exists():
        raise FileNotFoundError(f"Auto-selected chat template does not exist: {template}")
    return template


def _template_name_for_model(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen" in lowered:
        return "qwen3.jinja"
    if "smollm" in lowered:
        return "smollm.jinja"
    if "gemma" in lowered:
        return "gemma3.jinja"
    raise ValueError(f"Cannot auto-select PostTrainBench chat template for model: {model_id}")


def _generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_gen_toks": args.max_tokens}
    if args.task in {"gpqamain", "humaneval"}:
        kwargs["temperature"] = 1.0
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.do_sample != "auto":
        kwargs["do_sample"] = args.do_sample == "true"
    return kwargs


def _generation_overrides_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": None if args.do_sample == "auto" else args.do_sample == "true",
    }


if __name__ == "__main__":
    main()
