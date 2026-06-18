#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/hcgu/hlv/api_judge
RESULT_ROOT=/data/hcgu/hlv/results/gpt5mini_healthbench_base
REPO=/data/hcgu/projects/PostTrainBench
PY=/data/hcgu/.conda/envs/lm-eval/bin/python

set -a
source "${ROOT}/chatanywhere.env"
set +a

export GPT5_JUDGE_MODEL=gpt-5-mini
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="/data/hcgu/.conda/envs/lm-eval/bin:${PATH}"

HEALTH_LIMIT="${HEALTH_LIMIT:--1}"
HEALTH_JUDGE_WORKERS="${HEALTH_JUDGE_WORKERS:-4}"
HEALTH_EXAMPLE_WORKERS="${HEALTH_EXAMPLE_WORKERS:-2}"
HEALTH_CRITERIA_WORKERS="${HEALTH_CRITERIA_WORKERS:-2}"
HEALTH_MAX_TOKENS="${HEALTH_MAX_TOKENS:-4096}"

mkdir -p "${RESULT_ROOT}/healthbench" "${ROOT}/logs"

models=(
  "Qwen3-1.7B-Base|/data/hcgu/posttrainbench_results/baseline_generation/qwen3_1_7b_base_675"
  "Qwen3-4B-Base|/data/hcgu/posttrainbench_results/baseline_generation/Qwen_Qwen3-4B-Base_693"
  "SmolLM3-3B-Base|/data/hcgu/posttrainbench_results/baseline_generation/HuggingFaceTB_SmolLM3-3B-Base_693"
  "gemma-3-4b-pt|/data/hcgu/posttrainbench_results/baseline_generation/google_gemma-3-4b-pt_693"
)

echo "judge_model=${GPT5_JUDGE_MODEL}"
echo "health_limit=${HEALTH_LIMIT}"
echo "judge_workers=${HEALTH_JUDGE_WORKERS}"
echo "example_workers=${HEALTH_EXAMPLE_WORKERS}"
echo "criteria_workers=${HEALTH_CRITERIA_WORKERS}"
echo "started=$(date --iso-8601=seconds)"

for item in "${models[@]}"; do
  IFS='|' read -r alias base_dir <<< "${item}"
  answer_file="${base_dir}/healthbench/healthbench_model_answer.jsonl"
  test -s "${answer_file}"

  echo "===== HealthBench ${alias} $(date --iso-8601=seconds) ====="
  "${PY}" "${ROOT}/health_gpt5_judge.py" \
    --repo "${REPO}" \
    --answer-file "${answer_file}" \
    --model-alias "${alias}" \
    --output-dir "${RESULT_ROOT}/healthbench" \
    --judge-model "${GPT5_JUDGE_MODEL}" \
    --judge-workers "${HEALTH_JUDGE_WORKERS}" \
    --example-workers "${HEALTH_EXAMPLE_WORKERS}" \
    --criteria-workers "${HEALTH_CRITERIA_WORKERS}" \
    --limit "${HEALTH_LIMIT}" \
    --max-tokens "${HEALTH_MAX_TOKENS}"
done

echo "===== DONE $(date --iso-8601=seconds) ====="
