from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataos.core.artifact import ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.eval.runner import run_eval_stage


def main() -> None:
    args = parse_args()
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
        producer="posttrainbench_gsm8k_eval",
    )
    artifact = run_eval_stage([model_artifact], sandbox, task=_Task(args, model_source))
    summary = dict(artifact[0].meta)
    protocol = json.loads((artifact[0].path / "protocol.json").read_text(encoding="utf-8"))
    if args.json_output_file:
        output = {
            "accuracy": summary.get("score"),
            "results": summary.get("results", {}),
            "artifact": str(artifact[0].path),
            "source_results_json": str(artifact[0].path / "results.json"),
            "source_protocol_json": str(artifact[0].path / "protocol.json"),
            "task_protocol": "gsm8k_10shot_seed42_chat_template_numeric_accuracy",
        }
        Path(args.json_output_file).write_text(
            json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    print("ARTIFACT", artifact[0].path)
    print(
        "SUMMARY",
        json.dumps(
            {
                "mode": summary.get("mode"),
                "model": summary.get("model"),
                "tasks": summary.get("tasks"),
                "limit": summary.get("limit"),
                "score": summary.get("score"),
                "results": summary.get("results"),
                "resolved": protocol.get("resolved"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DataOS lm-eval on PostTrainBench GSM8K.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="posttrainbench-gsm8k-model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-gen-toks", type=int, default=4000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--json-output-file", default=None)
    parser.add_argument(
        "--chat-template",
        default="auto",
        help=(
            "PostTrainBench chat template path. Use 'auto' to select by model family, "
            "or 'none' to leave the backend/tokenizer default unchanged."
        ),
    )
    return parser.parse_args()


class _Task:
    def __init__(self, args: argparse.Namespace, model_source: str) -> None:
        model_args: dict[str, Any] = {
            "pretrained": model_source,
            "trust_remote_code": True,
            "dtype": "auto",
        }
        if args.backend == "vllm":
            model_args.update(
                {
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_model_len": args.max_model_len,
                }
            )
            chat_template = _resolve_chat_template(args.chat_template, model_source, args.model_name)
            if chat_template is not None:
                model_args["chat_template"] = str(chat_template)
        else:
            chat_template = None
        self.eval_config = {
            "mode": "lm_eval_native",
            "model": args.backend,
            "model_args": model_args,
            "tasks": ["posttrainbench_gsm8k"],
            "include_path": str(PROJECT_ROOT / "dataos" / "eval_tasks" / "posttrainbench"),
            "system_instruction": _posttrainbench_gsm8k_system_instruction(),
            "apply_chat_template": True,
            "fewshot_as_multiturn": False,
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "limit": _normalize_limit(args.limit),
            "gen_kwargs": {"max_gen_toks": args.max_gen_toks},
            "log_samples": True,
            "metadata": {
                "benchmark": "PostTrainBench",
                "task_protocol": "gsm8k_10shot_seed42_chat_template_numeric_accuracy",
                "chat_template": str(chat_template) if chat_template is not None else None,
                "display_scale": "accuracy_fraction; multiply by 100 for scores.json",
            },
        }


def _normalize_limit(limit: int | None) -> int | None:
    if limit is None or limit < 0:
        return None
    return limit


def _posttrainbench_gsm8k_system_instruction() -> str:
    train = list(load_dataset("openai/gsm8k", "main", split="train"))
    random.Random(42).shuffle(train)
    return "\n\n".join(_sample_to_fewshot(record) for record in train[:10])


def _sample_to_fewshot(record: dict[str, str]) -> str:
    reasoning, target = record["answer"].rsplit("####", maxsplit=1)
    return (
        f"{record['question']}\n\nReasoning:\n"
        f"{reasoning.strip()}\n\n"
        f"ANSWER: {target.strip()}"
    )


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


if __name__ == "__main__":
    main()
