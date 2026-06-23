#!/usr/bin/env python3
"""Convert protocol-eval generation output into the answer-file each Bailian judge expects.

The Slurm GPU job runs ``run_posttrainbench_protocol_eval`` which, for the two judge benches,
only GENERATES answers (no internet on compute nodes) and writes a simplified predictions file:

  arenahardwriting -> arena_predictions.jsonl   {uid, category, prompt, answer, baseline_*}
  healthbench      -> healthbench_predictions.jsonl  {prompt_id, prompt, answer, rubrics}

The vendored Bailian judges expect a different shape:

  arena_bailian_judge   -> native model_answer:  {uid, messages:[user, assistant{content:{answer}}]}
                           (baselines + categories are reloaded by judge_answers from the vendored
                            task DATA_PATH, so only uid + the candidate answer are needed here)
  health_bailian_judge  -> {example_id, response}

This converter bridges the two so the gen/judge split stays self-contained.
"""
from __future__ import annotations

import argparse
import json


def convert(bench: str, in_path: str, out_path: str, model_alias: str) -> int:
    n = 0
    with open(in_path, encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if bench == "arenahardwriting":
                # arena's judge filters battles by record["model"] == model_alias and reads the
                # candidate text from messages[-1].content.answer — both must be present.
                rec = {
                    "uid": r["uid"],
                    "model": model_alias,
                    "messages": [
                        {"role": "user", "content": r.get("prompt", "")},
                        {"role": "assistant", "content": {"answer": r["answer"]}},
                    ],
                }
            elif bench == "healthbench":
                rec = {
                    "example_id": r.get("example_id", r.get("prompt_id")),
                    "response": r.get("response", r.get("answer", "")),
                }
            else:
                raise ValueError(f"unknown judge bench: {bench}")
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", required=True, choices=["arenahardwriting", "healthbench"])
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--model-alias", default="candidate",
                   help="written into each arena record's 'model' field; must match the judge's --model-alias")
    args = p.parse_args()
    n = convert(args.bench, args.in_path, args.out_path, args.model_alias)
    print(f"[convert] {args.bench}: {n} records -> {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
