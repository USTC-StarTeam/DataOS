"""Run-scoped Curation-Bench suite coordinator, executed in the pinned provider venv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _prompt_text(task: str, iterations: int, profile: str) -> str:
    return (
        "---\n"
        f"tasks:\n  - {task}\n"
        f"iterations: {iterations}\n"
        f"profile: {profile}\n"
        "---\n"
        "DataOS owns planning and review; the harness owns every frozen iteration.\n"
    )


def _ensure(args):
    from benchmark.tools.suite import SuiteController

    project_root = Path(args.project_root)
    suite_dir = project_root / "suite"
    state_path = suite_dir / "suite_state.json"
    project_root.mkdir(parents=True, exist_ok=True)
    prompt_path = project_root / "run_prompt.md"
    expected_prompt = _prompt_text(args.task, args.iterations, args.profile)
    if prompt_path.exists() and prompt_path.read_text(encoding="utf-8") != expected_prompt:
        raise RuntimeError(f"existing suite prompt does not match this run: {prompt_path}")
    if not prompt_path.exists():
        prompt_path.write_text(expected_prompt, encoding="utf-8")
    if state_path.exists():
        controller = SuiteController.from_state_file(state_path)
    else:
        controller = SuiteController.from_prompt(
            prompt_path,
            profile_path=args.profile,
            output_root=project_root,
            suite_stamp="suite",
            agent="dataos",
            new_suite=True,
        )
    pairs = [(target.task_id, target.iteration) for target in controller.targets]
    expected = [(args.task, iteration) for iteration in range(1, args.iterations + 1)]
    if pairs != expected:
        raise RuntimeError(f"suite target queue mismatch: expected {expected}, got {pairs}")
    (project_root / "suite_ref.json").write_text(
        json.dumps(
            {
                "suite_dir": str(controller.suite_dir),
                "task": args.task,
                "iterations": args.iterations,
                "profile": args.profile,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return controller


def _target_result(controller, target) -> dict:
    return {
        "status": "target_activated",
        "suite_dir": str(controller.suite_dir),
        "target_key": target.key,
        "task_id": target.task_id,
        "iteration": target.iteration,
        "target_state": target.state,
        "run_dir": str(target.run_dir),
        "default_submission_path": str(target.run_dir / "curated" / "final_submission"),
    }


def claim(args) -> dict:
    from benchmark.tools.suite import ACTIVE, COMPLETED, FAILED, PENDING

    controller = _ensure(args)
    expected = args.expected_iteration
    target = controller.targets[expected - 1]
    if target.state in (ACTIVE, COMPLETED):
        return _target_result(controller, target)
    if target.state == FAILED:
        raise RuntimeError(f"cannot resume failed official target: {target.key}")
    if target.state != PENDING:
        raise RuntimeError(f"unknown target state for {target.key}: {target.state}")

    active = controller.active_target
    if active is not None and active.state == ACTIVE:
        if active.iteration >= expected:
            raise RuntimeError(
                f"cannot claim iteration {expected}; active target is {active.key}"
            )
        # A later durable DataOS node proves the prior attempt ended. Disable the controller's
        # crash-resume branch so its own begin_next_target transition records that target FAILED.
        controller._resumed_active = False
    result = controller.begin_next_target()
    if result.get("status") != "target_activated" or result.get("iteration") != expected:
        raise RuntimeError(f"official suite activated the wrong target: {result}")
    result["suite_dir"] = str(controller.suite_dir)
    result["target_state"] = ACTIVE
    return result


def finalize(args) -> dict:
    from benchmark.tools.suite import ACTIVE, PENDING

    controller = _ensure(args)
    if any(target.state == PENDING for target in controller.targets):
        raise RuntimeError("cannot finalize a suite with pending iterations")
    active = controller.active_target
    if active is not None and active.state == ACTIVE:
        controller._resumed_active = False
    result = controller.begin_next_target()
    if result.get("status") != "suite_completed":
        raise RuntimeError(f"official suite did not complete: {result}")
    result["suite_dir"] = str(controller.suite_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("claim", "finalize"))
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--expected-iteration", type=int)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.action == "claim":
        if args.expected_iteration is None or not 1 <= args.expected_iteration <= args.iterations:
            parser.error("claim requires --expected-iteration within the suite")
        result = claim(args)
    else:
        result = finalize(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
