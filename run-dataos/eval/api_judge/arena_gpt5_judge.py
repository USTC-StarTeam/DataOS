#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


def load_jsonl_by_uid(path: Path) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                answers[record["uid"]] = record
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge existing ArenaHard-Writing answers with GPT-5.")
    parser.add_argument("--repo", default="/data/hcgu/projects/PostTrainBench")
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default=os.environ.get("GPT5_JUDGE_MODEL", "gpt-5"))
    parser.add_argument("--judge-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    repo = Path(args.repo)
    task_dir = repo / "src/eval/tasks/arenahardwriting"
    sys.path.insert(0, str(task_dir))
    os.chdir(task_dir)

    import evaluate as arena_eval  # type: ignore

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    def call_openai_compatible(messages: list[dict[str, Any]]) -> dict[str, str] | None:
        request_kwargs: dict[str, Any] = {
            "model": args.judge_model,
            "messages": messages,
            "max_tokens": args.max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(arena_eval.API_MAX_RETRY):
            try:
                completion = client.chat.completions.create(**request_kwargs)
                content = completion.choices[0].message.content or ""
                if not content.strip():
                    raise RuntimeError("empty judge response")
                return {"answer": content}
            except Exception as err:  # noqa: BLE001
                last_error = err
                wait_time = arena_eval.API_RETRY_SLEEP * (2**attempt)
                print(f"[judge] GPT-5 API error ({type(err).__name__}): {err}. Retry in {wait_time}s.", flush=True)
                time.sleep(wait_time)
        print(f"[judge] Exhausted retries; last_error={last_error}", flush=True)
        return None

    arena_eval.JUDGE_MODEL = args.judge_model
    arena_eval.REASONING_EFFORT = None
    arena_eval.JUDGE_MAX_COMPLETION = args.max_tokens
    arena_eval.call_openai = call_openai_compatible

    run_args = argparse.Namespace(
        model_alias=args.model_alias,
        limit=args.limit,
        judge_workers=args.judge_workers,
        store_outputs=False,
    )
    candidate_answers = load_jsonl_by_uid(Path(args.answer_file))
    print(f"[data] loaded {len(candidate_answers)} answers for {args.model_alias}", flush=True)

    _, judgments = arena_eval.judge_answers(run_args, candidate_answers)
    metrics = arena_eval.summarize_results(args.model_alias, judgments)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judgment_path = output_dir / f"{args.model_alias}.judgments.jsonl"
    judged = [record for record in judgments if record is not None]
    with judgment_path.open("w", encoding="utf-8") as f:
        for record in judged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = metrics or {"accuracy": 0.0, "stderr": 0.0}
    metrics.update(
        {
            "benchmark": "arenahardwriting",
            "model": args.model_alias,
            "judge_model": args.judge_model,
            "limit": args.limit,
            "n_answers_total": len(candidate_answers),
            "n_judged": len(judged),
            "judgment_file": str(judgment_path),
        }
    )
    metrics_path = output_dir / f"{args.model_alias}.metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
