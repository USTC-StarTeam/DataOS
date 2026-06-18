#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/hcgu/hlv/api_judge
RESULT_ROOT=/data/hcgu/hlv/results/bailian_api_judge_qwen35plus
REPO=/data/hcgu/projects/PostTrainBench
PY=/data/hcgu/.conda/envs/lm-eval/bin/python

set -a
source "${ROOT}/dashscope.env"
set +a

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="/data/hcgu/.conda/envs/lm-eval/bin:${PATH}"

mkdir -p "${RESULT_ROOT}/arenahardwriting" "${RESULT_ROOT}/healthbench" "${ROOT}/logs"

models=(
  "Qwen3-1.7B-Base|/data/hcgu/posttrainbench_results/baseline_generation/qwen3_1_7b_base_675"
  "Qwen3-4B-Base|/data/hcgu/posttrainbench_results/baseline_generation/Qwen_Qwen3-4B-Base_693"
  "SmolLM3-3B-Base|/data/hcgu/posttrainbench_results/baseline_generation/HuggingFaceTB_SmolLM3-3B-Base_693"
  "gemma-3-4b-pt|/data/hcgu/posttrainbench_results/baseline_generation/google_gemma-3-4b-pt_693"
)

for item in "${models[@]}"; do
  IFS='|' read -r alias base_dir <<< "${item}"
  echo "===== ArenaHard ${alias} $(date --iso-8601=seconds) ====="
  "${PY}" "${ROOT}/arena_bailian_judge.py" \
    --repo "${REPO}" \
    --answer-file "${base_dir}/arenahardwriting/arenahardwriting_model_answer.jsonl" \
    --model-alias "${alias}" \
    --output-dir "${RESULT_ROOT}/arenahardwriting" \
    --judge-workers "${ARENA_JUDGE_WORKERS:-16}" \
    --limit "${LIMIT:--1}"

  echo "===== HealthBench ${alias} $(date --iso-8601=seconds) ====="
  "${PY}" "${ROOT}/health_bailian_judge.py" \
    --repo "${REPO}" \
    --answer-file "${base_dir}/healthbench/healthbench_model_answer.jsonl" \
    --model-alias "${alias}" \
    --output-dir "${RESULT_ROOT}/healthbench" \
    --judge-workers "${HEALTH_JUDGE_WORKERS:-16}" \
    --limit "${LIMIT:--1}"
done

echo "===== DONE $(date --iso-8601=seconds) ====="
