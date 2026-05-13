from __future__ import annotations

import json
from typing import Any

PLAN_MAX_STEPS = 6
REVIEW_REQUIRED_FIELDS = ["summary", "issues", "next_hypothesis", "recommended_changes"]

PLAN_RULES = [
    "Only choose operators that appear in the provided registry.",
    f"Return valid JSON only, with no prose outside JSON, and no more than {PLAN_MAX_STEPS} steps.",
    "Only plan data operators. Training and evaluation run automatically as fixed runtime stages after the operator plan.",
    "Do not invent new operators.",
    "Do not ask to modify code, execute shell commands, or generate a new operator.",
    "Treat all datasets as Hugging Face datasets artifacts, never JSONL.",
    "Use dataset summaries based on hf_object_type, splits, num_rows, columns, and features.",
    "Use only parameters that appear in the operator schema; fill required parameters and prefer defaults when the task does not constrain them.",
    "Operators always consume the latest upstream dataset artifact. Plan a sequence that is executable in order.",
    "If the latest upstream dataset does not already have Alpaca-style columns like instruction/input/output, use a mapper such as map_to_alpaca before filters or evaluators that rely on those columns.",
    "When a filter or selector references specific columns, make sure those columns exist on the dataset produced by the previous step.",
]

REVIEW_RULES = [
    "Return JSON only.",
    "Summarize the experiment, surface issues, and propose the next hypothesis.",
    "Do not request code changes or shell commands from the LLM.",
]


def build_plan_system_prompt(max_steps: int = PLAN_MAX_STEPS) -> str:
    joined_rules = "\n".join(f"- {rule}" for rule in PLAN_RULES)
    return (
        "You are the planning component for DataOS.\n"
        f"The runtime can execute at most {max_steps} operators.\n"
        "Follow these rules exactly:\n"
        f"{joined_rules}\n"
        "Output schema:\n"
        '{\n'
        '  "steps": [{"operator_name": "map_to_alpaca", "params": {"keep_other_columns": false}}, {"operator_name": "filter_exact_dedup", "params": {"keys": ["instruction"]}}],\n'
        '  "reasoning_summary": "short string",\n'
        '  "expected_gain": "short string"\n'
        '}'
    )


def build_plan_user_prompt(
    task_record: dict[str, Any],
    initial_artifacts: list[dict[str, Any]],
    history: list[dict[str, Any]],
    operators: list[dict[str, str]],
) -> str:
    payload = {
        "task": task_record,
        "initial_artifacts": initial_artifacts,
        "history": history,
        "available_operators": operators,
        "execution_note": "Each operator runs against the latest upstream dataset artifact, which is initially the last dataset in initial_artifacts.",
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def build_review_system_prompt() -> str:
    joined_rules = "\n".join(f"- {rule}" for rule in REVIEW_RULES)
    return (
        "You are the review component for DataOS.\n"
        "Follow these rules exactly:\n"
        f"{joined_rules}\n"
        "Output schema:\n"
        '{\n'
        '  "summary": "short string",\n'
        '  "issues": ["issue 1"],\n'
        '  "next_hypothesis": "short string",\n'
        '  "recommended_changes": ["change 1"]\n'
        '}'
    )


def build_review_user_prompt(
    task_record: dict[str, Any],
    experiment_record: dict[str, Any],
    operators: list[dict[str, str]],
) -> str:
    payload = {
        "task": task_record,
        "experiment": experiment_record,
        "available_operators": operators,
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def validate_plan_payload(
    payload: dict[str, Any],
    operator_lookup: dict[str, Any],
    max_steps: int = PLAN_MAX_STEPS,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Plan payload must be a JSON object")
    if "steps" not in payload or "reasoning_summary" not in payload or "expected_gain" not in payload:
        raise ValueError("Plan payload missing required fields")
    steps = payload["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("Plan payload requires a non-empty steps list")
    if len(steps) > max_steps:
        raise ValueError(f"Plan exceeds max steps: {len(steps)} > {max_steps}")
    normalized_steps: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Each plan step must be an object")
        operator_name = step.get("operator_name")
        if operator_name not in operator_lookup:
            raise ValueError(f"Plan uses unregistered operator: {operator_name}")
        params = step.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Plan step params must be an object")
        normalized_steps.append(
            {
                "operator_name": operator_name,
                "params": operator_lookup[operator_name].normalize_params(params),
            }
        )
    return {
        "steps": normalized_steps,
        "reasoning_summary": str(payload["reasoning_summary"]),
        "expected_gain": str(payload["expected_gain"]),
    }


def normalize_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Review payload must be a JSON object")
    missing = [field_name for field_name in REVIEW_REQUIRED_FIELDS if field_name not in payload]
    if missing:
        raise ValueError(f"Review payload missing required fields: {', '.join(missing)}")
    issues = payload["issues"]
    recommended_changes = payload["recommended_changes"]
    if not isinstance(issues, list):
        raise ValueError("Review issues must be a list")
    if not isinstance(recommended_changes, list):
        raise ValueError("Review recommended_changes must be a list")
    return {
        "summary": str(payload["summary"]),
        "issues": [str(item) for item in issues],
        "next_hypothesis": str(payload["next_hypothesis"]),
        "recommended_changes": [str(item) for item in recommended_changes],
    }
