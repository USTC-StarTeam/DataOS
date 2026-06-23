#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


def load_answer_map(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            answers[record["example_id"]] = record["response"]
    return answers


def parse_json_to_dict(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge existing HealthBench answers with Bailian.")
    parser.add_argument("--repo", default="/data/hcgu/projects/PostTrainBench")
    # In-branch vendored layout: point straight at resources/code/posttrainbench
    # (the official `src/eval`). Falls back to <repo>/src/eval for the legacy checkout.
    parser.add_argument("--ptb-eval-src", default=os.environ.get("POSTTRAINBENCH_EVAL_SRC", ""))
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default=os.environ.get("BAILIAN_JUDGE_MODEL", "qwen3.5-plus"))
    parser.add_argument("--judge-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    eval_src = Path(args.ptb_eval_src) if args.ptb_eval_src else Path(args.repo) / "src/eval"
    task_dir = eval_src / "tasks/healthbench"
    sys.path.insert(0, str(task_dir))
    os.chdir(task_dir)

    from evaluation_code import grader  # type: ignore
    from evaluation_code.data_loader import load_healthbench  # type: ignore
    from evaluation_code.scoring import aggregate_scores  # type: ignore

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )

    def custom_grade_criterion(
        conversation: list[dict[str, Any]],
        model_response: str,
        criterion: Any,
        grader_model: str = args.judge_model,
        client: OpenAI | None = None,
        max_retries: int = 5,
    ) -> Any:
        conv_formatted = grader.format_conversation_for_grader(conversation, model_response)
        rubric_str = grader.format_rubric_item(criterion.criterion, criterion.points)
        prompt = grader.GRADER_TEMPLATE.replace("<<conversation>>", conv_formatted).replace("<<rubric_item>>", rubric_str)
        last_error: Exception | None = None
        active_client = client or globals()["client"]
        for attempt in range(max_retries):
            try:
                grader._acquire_api_slot()
                try:
                    response = active_client.chat.completions.create(
                        model=grader_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        response_format={"type": "json_object"},
                        extra_body={"enable_thinking": False},
                    )
                finally:
                    grader._release_api_slot()
                content = response.choices[0].message.content or ""
                parsed = parse_json_to_dict(content)
                label = parsed.get("criteria_met")
                if label is True or label is False:
                    return grader.GradingResult(
                        criterion_id=criterion.criterion_id,
                        criteria_met=bool(label),
                        explanation=parsed.get("explanation", "No explanation provided"),
                        weighted_score=criterion.points if bool(label) else 0,
                    )
                print(f"[judge] bad JSON on attempt {attempt + 1}: {content[:200]}", flush=True)
            except Exception as err:  # noqa: BLE001
                last_error = err
                print(f"[judge] Bailian API error ({type(err).__name__}): {err}. Retry.", flush=True)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"All retries failed for criterion {criterion.criterion_id}") from last_error

    grader.get_client = lambda: client
    grader.grade_criterion = custom_grade_criterion

    examples = load_healthbench()
    random.Random(42).shuffle(examples)
    if args.limit != -1:
        examples = examples[: args.limit]

    answer_map = load_answer_map(Path(args.answer_file))
    missing = [example.example_id for example in examples if example.example_id not in answer_map]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} HealthBench answers, first={missing[:3]}")
    responses = [answer_map[example.example_id] for example in examples]
    total_calls = sum(len(example.rubric_criteria) for example in examples)
    print(f"[data] examples={len(examples)} grader_calls={total_calls} model={args.model_alias}", flush=True)

    results = grader.grade_examples_parallel(
        examples=examples,
        responses=responses,
        grader_model=args.judge_model,
        example_workers=min(4, len(examples)),
        criteria_workers=8,
        max_concurrent_requests=args.judge_workers,
    )
    benchmark_result = aggregate_scores(results, examples)
    metrics: dict[str, Any] = {
        "benchmark": "healthbench",
        "model": args.model_alias,
        "judge_model": args.judge_model,
        "accuracy": benchmark_result.accuracy,
        "stderr": benchmark_result.stderr,
        "n_examples": benchmark_result.n_examples,
        "total_grader_calls": benchmark_result.total_grader_calls,
        "by_theme": benchmark_result.by_theme,
        "by_axis": benchmark_result.by_axis,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / f"{args.model_alias}.judgments.jsonl"
    with judgments_path.open("w", encoding="utf-8") as f:
        for example, response, result in zip(examples, responses, results):
            f.write(
                json.dumps(
                    {
                        "example_id": example.example_id,
                        "model": args.model_alias,
                        "response": response,
                        "total_score": result.total_score,
                        "max_possible_score": result.max_possible_score,
                        "normalized_score": result.normalized_score,
                        "grading_results": [r.__dict__ for r in result.grading_results],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    metrics["judgment_file"] = str(judgments_path)
    metrics_path = output_dir / f"{args.model_alias}.metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
