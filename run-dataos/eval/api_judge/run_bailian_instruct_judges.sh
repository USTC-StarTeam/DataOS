#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/hcgu/hlv/api_judge
RESULT_ROOT=/data/hcgu/hlv/results/bailian_api_judge_qwen35plus_instruct
GEN_ROOT=/data/hcgu/hlv/results/instruct_generation/zero_shot_1021
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
  "Qwen3-1.7B|${GEN_ROOT}/Qwen_Qwen3-1.7B"
  "Qwen3-4B|${GEN_ROOT}/Qwen_Qwen3-4B"
  "SmolLM3-3B|${GEN_ROOT}/HuggingFaceTB_SmolLM3-3B"
  "gemma-3-4b-it|${GEN_ROOT}/google_gemma-3-4b-it"
)

for item in "${models[@]}"; do
  IFS='|' read -r alias base_dir <<< "${item}"

  arena_answer="${base_dir}/arenahardwriting/arenahardwriting_model_answer.jsonl"
  health_answer="${base_dir}/healthbench/healthbench_model_answer.jsonl"
  test -s "${arena_answer}"
  test -s "${health_answer}"

  echo "===== ArenaHard ${alias} $(date --iso-8601=seconds) ====="
  "${PY}" "${ROOT}/arena_bailian_judge.py" \
    --repo "${REPO}" \
    --answer-file "${arena_answer}" \
    --model-alias "${alias}" \
    --output-dir "${RESULT_ROOT}/arenahardwriting" \
    --judge-workers "${ARENA_JUDGE_WORKERS:-16}" \
    --limit "${LIMIT:--1}"

  echo "===== HealthBench ${alias} $(date --iso-8601=seconds) ====="
  "${PY}" "${ROOT}/health_bailian_judge.py" \
    --repo "${REPO}" \
    --answer-file "${health_answer}" \
    --model-alias "${alias}" \
    --output-dir "${RESULT_ROOT}/healthbench" \
    --judge-workers "${HEALTH_JUDGE_WORKERS:-16}" \
    --limit "${LIMIT:--1}"
done

echo "===== DONE $(date --iso-8601=seconds) ====="
