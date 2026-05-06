from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.agent import LLMPlanningAgent, PlanStep
from dataos.core.artifact import EvalArtifact, ModelArtifact, make_artifact_id
from dataos.core.registry import OperatorRegistry
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.llm.client import LLMClient
from dataos.llm.openai_compat import OpenAICompatLLMClient
from dataos.operators.dedup import DedupOperator
from dataos.operators.sample import SampleOperator
from dataos.prompts.agent_rules import validate_plan_payload
from dataos.runtime import runner as runtime_runner
from dataos.runtime.runner import run_task


def _require_live_llm_env() -> dict[str, str]:
    base_url = os.environ.get("DASHSCOPE_BASE_URL") or os.environ.get("STARTEAM_BASE_URL")
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("STARTEAM_API_KEY")
    model = os.environ.get("DASHSCOPE_MODEL") or os.environ.get("STARTEAM_MODEL")
    missing = [
        name
        for name, value in {
            "DASHSCOPE_BASE_URL/STARTEAM_BASE_URL": base_url,
            "DASHSCOPE_API_KEY/STARTEAM_API_KEY": api_key,
            "DASHSCOPE_MODEL/STARTEAM_MODEL": model,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing live LLM environment variables: {', '.join(missing)}")
    return {"base_url": str(base_url), "api_key": str(api_key), "model": str(model)}


def _load_gsm8k_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < 8:
        raise ValueError("gsm8k json must be a JSON array with at least 8 rows")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"gsm8k row {index} is not an object")
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        normalized.append({"question": question, "answer": answer})
    if len(normalized) < 8:
        raise ValueError("gsm8k json did not contain enough usable rows")
    return normalized


def _build_dataset(gsm8k_rows: list[dict[str, Any]], *, train_rows: int, test_rows: int) -> DatasetDict:
    unique_rows = gsm8k_rows[: max(train_rows + test_rows, 8)]
    train_slice = unique_rows[:train_rows]
    test_slice = unique_rows[train_rows : train_rows + test_rows]
    train_records = [
        {
            "instruction": row["question"],
            "input": "",
            "response": row["answer"],
            "source_split": "gsm8k-train-subset",
        }
        for row in train_slice
    ]
    duplicate = dict(train_records[0])
    duplicate["source_split"] = "gsm8k-train-duplicate"
    train_records.insert(1, duplicate)
    test_records = [
        {
            "instruction": row["question"],
            "input": "",
            "response": row["answer"],
            "source_split": "gsm8k-test-subset",
        }
        for row in test_slice
    ]
    return DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "test": Dataset.from_list(test_records),
        }
    )


def _build_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    registry.register(DedupOperator())
    registry.register(SampleOperator())
    return registry


def _safe_plan_payload() -> dict[str, Any]:
    return {
        "steps": [
            {"operator_name": "dedup", "params": {"keys": ["instruction", "response"]}},
            {"operator_name": "sample", "params": {"max_train": 6, "max_test": 3}},
        ],
        "reasoning_summary": "Use dedup to remove the intentional duplicate and sample to keep the GSM8K subset compact.",
        "expected_gain": "Confines the live planner to the two supported preprocessing operators before stub runtime stages.",
    }


class GuardedOpenAICompatLLMClient(LLMClient):
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.inner = OpenAICompatLLMClient(api_key=api_key, base_url=base_url, model=model)
        self.last_plan_raw_payload: dict[str, Any] | None = None
        self.last_plan_used_payload: dict[str, Any] | None = None
        self.last_review_raw_payload: dict[str, Any] | None = None

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload = self.inner.generate_json(system_prompt, user_prompt, temperature=temperature)
        if system_prompt.startswith("You are the planning component for DataOS."):
            self.last_plan_raw_payload = payload
            try:
                normalized = validate_plan_payload(payload, {"dedup", "sample"}, max_steps=2)
            except Exception:
                normalized = _safe_plan_payload()
            self.last_plan_used_payload = normalized
            return normalized
        self.last_review_raw_payload = payload
        return payload


def _patched_eval_stage(
    artifacts: list[Any],
    sandbox: Sandbox,
    *,
    task: Any,
    experiment: Any | None = None,
    **_unused: Any,
) -> list[EvalArtifact]:
    del task, experiment, _unused
    model_artifact: ModelArtifact | None = None
    for artifact in reversed(artifacts):
        if isinstance(artifact, ModelArtifact):
            model_artifact = artifact
            break
    if model_artifact is None:
        raise ValueError("stub eval requires a ModelArtifact input")

    artifact_id = make_artifact_id("eval")
    eval_dir = sandbox.artifact_dir("eval", artifact_id)
    payload = {
        "backend": "stub",
        "tasks": ["gsm8k_stub"],
        "source_model_id": model_artifact.artifact_id,
        "score": None,
        "headline_score": None,
        "note": "Live planner/review only; training and evaluation are stubbed in this demo.",
    }
    (eval_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (eval_dir / "lm_eval_results.json").write_text(
        json.dumps({"backend": "stub", "tasks": ["gsm8k_stub"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifact = EvalArtifact(
        artifact_id=artifact_id,
        name=f"eval-{model_artifact.artifact_id}",
        path=eval_dir,
        parent_ids=[model_artifact.artifact_id],
        producer="stub_eval",
        meta=payload,
    )
    artifact.save_meta()
    sandbox.log(
        "stub_eval",
        f"model={model_artifact.artifact_id}\neval={artifact.artifact_id}\ntask=gsm8k_stub\n",
    )
    return [artifact]


def _safe_fallback_plan(self: LLMPlanningAgent) -> list[PlanStep]:
    return [
        PlanStep("dedup", {"keys": ["instruction", "response"]}),
        PlanStep("sample", {"max_train": 6, "max_test": 3}),
    ]


def run_demo(
    *,
    workspace: Path,
    gsm8k_json: Path,
    overwrite: bool,
    train_rows: int,
    test_rows: int,
) -> dict[str, object]:
    env = _require_live_llm_env()
    if overwrite and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset = _build_dataset(_load_gsm8k_rows(gsm8k_json), train_rows=train_rows, test_rows=test_rows)
    dataset_path = workspace / "gsm8k_live_llm_dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset.save_to_disk(str(dataset_path))

    task_template = TaskTemplate.from_dict(
        {
            "task_id": "live-llm-runtime-demo",
            "name": "Live LLM Runtime Demo",
            "goal": "Use a live OpenAI-compatible planner to choose between dedup and sample for a real GSM8K subset before stub runtime stages.",
            "input_description": "A real GSM8K-derived DatasetDict with one intentional duplicate training pair.",
            "output_description": "A plan/review trace plus stub model/eval artifacts.",
            "constraints": {
                "dataset_format": "hf_datasets",
                "execution_mode": "fixed_runtime",
                "llm_role": ["plan", "review"],
            },
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["dedup", "sample"],
            "train_config": {"mode": "stub"},
            "eval_config": {"mode": "stub", "tasks": ["gsm8k_stub"]},
        }
    )

    llm_client = GuardedOpenAICompatLLMClient(
        api_key=env["api_key"],
        base_url=env["base_url"],
        model=env["model"],
    )

    original_build_default_registry = runtime_runner.build_default_registry
    original_run_eval_stage = runtime_runner.run_eval_stage
    original_fallback_plan = LLMPlanningAgent._fallback_plan
    runtime_runner.build_default_registry = _build_registry
    runtime_runner.run_eval_stage = _patched_eval_stage
    LLMPlanningAgent._fallback_plan = _safe_fallback_plan
    try:
        result = run_task(
            task_template=task_template,
            dataset_path=dataset_path,
            workspace_dir=workspace / "runtime_workspace",
            llm_client=llm_client,
        )
    finally:
        runtime_runner.build_default_registry = original_build_default_registry
        runtime_runner.run_eval_stage = original_run_eval_stage
        LLMPlanningAgent._fallback_plan = original_fallback_plan

    steps = result.experiment.to_record().get("steps", [])
    summary = {
        "workspace": str(workspace),
        "gsm8k_json": str(gsm8k_json),
        "dataset_path": str(dataset_path),
        "planner_model": env["model"],
        "allowed_operator_names": ["dedup", "sample"],
        "experiment_id": result.experiment.exp_id,
        "experiment_status": result.experiment.status,
        "review": result.review,
        "llm_plan_raw_payload": llm_client.last_plan_raw_payload,
        "llm_plan_used_payload": llm_client.last_plan_used_payload,
        "llm_review_raw_payload": llm_client.last_review_raw_payload,
        "agent_plan_payload": result.agent.last_plan_payload,
        "steps": steps,
        "executed_operator_names": [step["operator_name"] for step in steps if step.get("step_type") == "operator"],
        "artifacts": [artifact.summary() for artifact in result.experiment.artifacts],
        "sandbox_root": str(result.experiment.sandbox.root),
        "plan_path": str(result.experiment.sandbox.root / "plan.json"),
        "review_path": str(result.experiment.sandbox.root / "review.json"),
        "report_path": str(result.experiment.sandbox.root / "report.json"),
        "archived_experiment_path": str(result.archived_experiment_artifact.path),
        "stub_train_enabled": True,
        "stub_eval_enabled": True,
    }
    summary_path = workspace / "live_llm_runtime_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DataOS run_task() with a live planner/reviewer on real GSM8K data and stub runtime stages.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--gsm8k-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-rows", type=int, default=7)
    parser.add_argument("--test-rows", type=int, default=3)
    args = parser.parse_args()

    summary = run_demo(
        workspace=args.workspace.expanduser().resolve(),
        gsm8k_json=args.gsm8k_json.expanduser().resolve(),
        overwrite=args.overwrite,
        train_rows=int(args.train_rows),
        test_rows=int(args.test_rows),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
