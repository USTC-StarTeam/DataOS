#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/hcgu/hlv/api_judge
RESULT_ROOT="${RESULT_ROOT:-/data/hcgu/hlv/results/gpt5_api_judge_smoke}"
REPO=/data/hcgu/projects/PostTrainBench
PY=/data/hcgu/.conda/envs/lm-eval/bin/python

set -a
source "${ROOT}/chatanywhere.env"
set +a

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="/data/hcgu/.conda/envs/lm-eval/bin:${PATH}"

ARENA_LIMIT="${ARENA_LIMIT:-20}"
HEALTH_LIMIT="${HEALTH_LIMIT:-5}"
ARENA_JUDGE_WORKERS="${ARENA_JUDGE_WORKERS:-1}"
HEALTH_JUDGE_WORKERS="${HEALTH_JUDGE_WORKERS:-1}"
ARENA_MAX_TOKENS="${ARENA_MAX_TOKENS:-8192}"
HEALTH_MAX_TOKENS="${HEALTH_MAX_TOKENS:-4096}"
JUDGE_MODEL="${GPT5_JUDGE_MODEL:-gpt-5}"

mkdir -p "${RESULT_ROOT}/arenahardwriting" "${RESULT_ROOT}/healthbench" "${ROOT}/logs"

models=(
  "Qwen3-1.7B-Base|/data/hcgu/posttrainbench_results/baseline_generation/qwen3_1_7b_base_675"
  "Qwen3-4B-Base|/data/hcgu/posttrainbench_results/baseline_generation/Qwen_Qwen3-4B-Base_693"
  "SmolLM3-3B-Base|/data/hcgu/posttrainbench_results/baseline_generation/HuggingFaceTB_SmolLM3-3B-Base_693"
  "gemma-3-4b-pt|/data/hcgu/posttrainbench_results/baseline_generation/google_gemma-3-4b-pt_693"
)

echo "judge_model=${JUDGE_MODEL}"
echo "arena_limit=${ARENA_LIMIT}"
echo "health_limit=${HEALTH_LIMIT}"
echo "started=$(date --iso-8601=seconds)"

for item in "${models[@]}"; do
  IFS='|' read -r alias base_dir <<< "${item}"

  echo "===== ArenaHard ${alias} $(date --iso-8601=seconds) ====="
  "${PY}" "${ROOT}/arena_gpt5_judge.py" \
    --repo "${REPO}" \
    --answer-file "${base_dir}/arenahardwriting/arenahardwriting_model_answer.jsonl" \
    --model-alias "${alias}" \
    --output-dir "${RESULT_ROOT}/arenahardwriting" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-workers "${ARENA_JUDGE_WORKERS}" \
    --limit "${ARENA_LIMIT}" \
    --max-tokens "${ARENA_MAX_TOKENS}"

  echo "===== HealthBench ${alias} $(date --iso-8601=seconds) ====="
  "${PY}" "${ROOT}/health_gpt5_judge.py" \
    --repo "${REPO}" \
    --answer-file "${base_dir}/healthbench/healthbench_model_answer.jsonl" \
    --model-alias "${alias}" \
    --output-dir "${RESULT_ROOT}/healthbench" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-workers "${HEALTH_JUDGE_WORKERS}" \
    --limit "${HEALTH_LIMIT}" \
    --max-tokens "${HEALTH_MAX_TOKENS}"
done

echo "===== DONE $(date --iso-8601=seconds) ====="
