"""Run Curation-Bench VLMEvalKit in offline-infer or login-node-judge mode."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import yaml


def _target(suite_dir: Path, model_path: Path) -> tuple[dict, Path, list[str], Path]:
    state_path = suite_dir / "suite_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    targets = state.get("targets") or []
    active_index = state.get("active_index")
    if not isinstance(active_index, int) or not 0 <= active_index < len(targets):
        raise ValueError(f"suite has no active target: {state_path}")
    target = targets[active_index]
    run_dir = Path(target["run_dir"])
    task = yaml.safe_load(Path(target["task_path"]).read_text(encoding="utf-8"))["task"]
    session = json.loads((run_dir / "session_state.json").read_text(encoding="utf-8"))
    expected_model = Path(session["finetune_model_path"])
    if model_path.resolve() != expected_model.resolve():
        raise ValueError(f"model path {model_path} != active target model {expected_model}")
    profile = yaml.safe_load(Path(state["profile_path"]).read_text(encoding="utf-8"))
    eval_data_dir = Path(profile["eval_data_dir"])
    return target, run_dir, list(task["target_evals"]), eval_data_dir


def _run(args: argparse.Namespace) -> dict:
    target, run_dir, benchmarks, eval_data_dir = _target(args.suite_dir, args.model_path)
    model_name = _model_name(args.harness_root, Path(target["task_path"]))
    if args.vlmeval_model and args.vlmeval_model != model_name:
        raise ValueError(f"eval model {args.vlmeval_model!r} != official model {model_name!r}")
    vlmeval = args.harness_root / "vendor" / "VLMEvalKit"
    command = [
        str(vlmeval / ".venv" / "bin" / "python"),
        str(vlmeval / "run.py"),
        "--model",
        model_name,
        "--model-path",
        str(args.model_path),
        "--data",
        *benchmarks,
        "--work-dir",
        str(run_dir / "eval" / "results"),
        "--mode",
        args.mode,
        "--reuse",
    ]
    if args.mode == "infer":
        command.append("--use-vllm")
        if model_name == "Qwen2.5-VL-3B-Instruct":
            command.insert(1, str(Path(__file__).with_name("vlmeval_entry.py")))
    else:
        if not args.judge_model:
            raise ValueError("remote judge requires --judge-model")
        if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_API_BASE"):
            raise ValueError("remote judge requires OPENAI_API_KEY and OPENAI_API_BASE")
        command.extend(
            [
                "--judge",
                args.judge_model,
                "--api-nproc",
                str(args.workers),
                "--retry",
                str(args.request_retries),
            ]
        )
        if args.timeout_seconds is not None:
            command.extend(["--judge-args", json.dumps({"timeout": args.timeout_seconds})])

    env = os.environ.copy()
    env.update(
        {
            "LMUData": str(eval_data_dir),
            "BENCHMARK_VLMEVAL_TRUST_LOCAL_TSV": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    # Upstream catches each benchmark exception and may exit zero after skipping all eight.
    # A process exit code alone must never complete an official iteration.
    validation = [
        str(vlmeval / ".venv" / "bin" / "python"),
        str(Path(__file__).with_name("validate_predictions.py")),
        "--results-dir", str(run_dir / "eval" / "results"),
        "--model", model_name, "--benchmarks", *benchmarks,
    ]
    if args.mode == "eval":
        subprocess.run(validation, cwd=vlmeval, env=env, check=True)
    subprocess.run(command, cwd=vlmeval, env=env, check=True)
    if args.mode == "infer":
        subprocess.run(validation, cwd=vlmeval, env=env, check=True)
    if args.mode == "eval":
        _validate_scores(run_dir / "eval" / "results", model_name, benchmarks)
    result = {
        "status": "completed",
        "results_dir": str(run_dir / "eval" / "results"),
        "error": None,
    }
    if args.mode == "eval":
        mark = (
            "from benchmark.tools.suite import SuiteController; "
            f"ctrl=SuiteController.from_state_file({str(args.suite_dir / 'suite_state.json')!r}); "
            f"ctrl.mark_target_completed({result!r})"
        )
        subprocess.run(
            [str(args.harness_root / ".venv" / "bin" / "python"), "-c", mark],
            cwd=args.harness_root,
            check=True,
        )
    result["target"] = target["key"]
    return result


def _validate_scores(results_dir: Path, model: str, benchmarks: list[str]) -> None:
    import runpy

    score_value = runpy.run_path(str(Path(__file__).with_name("collect.py")))["_score"]
    config = yaml.safe_load(Path(__file__).with_name("task.yaml").read_text())["native"]
    results = json.loads((results_dir / "results.json").read_text())
    if set(results) != {model} or not isinstance(results[model], dict):
        raise ValueError("evaluation results do not match the official model")
    for benchmark in benchmarks:
        value = score_value(benchmark, results[model].get(benchmark), config["selectors"][benchmark])
        maximum = float(config["normalization"][benchmark])
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"invalid score for {benchmark}: {value}")


def _model_name(harness_root: Path, task_path: Path) -> str:
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))["task"]
    return subprocess.check_output(
        [
            str(harness_root / ".venv" / "bin" / "python"),
            "-c",
            "import sys; from benchmark.core.model_registry import get_model; "
            "print(get_model(sys.argv[1]).vlmeval_model_name)",
            task["model_key"],
        ],
        cwd=harness_root,
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["infer", "judge"])
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vlmeval-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--request-retries", type=int, default=1)
    args = parser.parse_args()
    args.mode = "eval" if args.mode == "judge" else args.mode
    try:
        result = _run(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports the frozen runner failure
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
