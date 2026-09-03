"""Run Curation-Bench VLMEvalKit in offline-infer or login-node-judge mode."""

from __future__ import annotations

import argparse
import json
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
    vlmeval = args.harness_root / "vendor" / "VLMEvalKit"
    command = [
        str(vlmeval / ".venv" / "bin" / "python"),
        str(vlmeval / "run.py"),
        "--model",
        args.vlmeval_model,
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
    else:
        if not args.judge_model:
            raise ValueError("remote judge requires --judge-model")
        if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_API_BASE"):
            raise ValueError("remote judge requires OPENAI_API_KEY and OPENAI_API_BASE")
        command.extend(["--judge", args.judge_model, "--api-nproc", str(args.workers)])

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
    subprocess.run(command, cwd=vlmeval, env=env, check=True)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["infer", "judge"])
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vlmeval-model", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--workers", type=int, default=32)
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
