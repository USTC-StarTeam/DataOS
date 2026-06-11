#!/usr/bin/env bash
# Frozen PostTrainBench harness runner — a DataOS `provider:"external"` command.
#
#   run_ptb.sh <bench> <model_path> <results_file> [limit]
#
# bench ∈ {bfcl, gsm8k, humaneval, aime2025, gpqamain, healthbench, arenahardwriting}
# limit: omit or -1 = full set. LoRA adapters are auto-merged (cached at <adapter>/_merged).
# Judge benches (healthbench/arenahardwriting) read the key from ~/.mimo_key (never an arg)
# and route the judge through the clash proxy; JUDGE_MODEL env may override (default MiMo).
#
# Frozen env (see PINNED_COMMITS.txt on the box): PostTrainBench d3496fa7 (+judge-env patch),
# inspect_evals bbc1e495, vllm 0.11.0 + transformers 4.57.3 (v5 breaks vllm's tokenizer API).
set -euo pipefail
BENCH=$1; MODEL=$2; OUT=$3; LIMIT=${4:--1}
PTB=${PTB_ROOT:-/data/hcgu/dev/PostTrainBench}
PY=$PTB/.venv/bin/python
TASK=$PTB/src/eval/tasks/$BENCH
HERE=$(cd "$(dirname "$0")" && pwd)
[ -d "$TASK" ] || { echo "unknown bench: $BENCH" >&2; exit 2; }

export PATH=$PTB/.venv/bin:$PATH \
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

# LoRA adapter -> merged model (vllm-based harnesses can't load bare adapters).
# Cache lives OUTSIDE the artifact dir (artifacts stay immutable); keyed by adapter path hash.
# Legacy in-artifact _merged dirs are still honored if already present.
if [ -f "$MODEL/adapter_config.json" ]; then
  if [ -f "$MODEL/_merged/config.json" ]; then
    MERGED="$MODEL/_merged"
  else
    CACHE=${PTB_MERGE_CACHE:-$HOME/.cache/dataos_ptb_merged}
    H=$(printf "%s" "$(cd "$MODEL" && pwd)" | md5sum | cut -c1-12)
    MERGED="$CACHE/$H"
    [ -f "$MERGED/config.json" ] || $PY "$HERE/merge_lora.py" "$MODEL" "$MERGED"
  fi
  MODEL="$MERGED"
fi

EXTRA=""
case "$BENCH" in
  healthbench|arenahardwriting)
    [ -s ~/.mimo_key ] || { echo "judge bench '$BENCH' needs ~/.mimo_key on this host" >&2; exit 3; }
    export HTTPS_PROXY=http://127.0.0.1:7890 \
      OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://token-plan-cn.xiaomimimo.com/v1} \
      JUDGE_MODEL=${JUDGE_MODEL:-mimo-v2.5-pro} \
      OPENAI_API_KEY=$(cat ~/.mimo_key)
    EXTRA="--judge-workers ${JUDGE_WORKERS:-2}"
    ;;
esac

RAW="$OUT.raw.json"
cd "$TASK"
$PY evaluate.py --model-path "$MODEL" --limit "$LIMIT" \
  --json-output-file "$RAW" --templates-dir "$PTB/src/eval/templates" $EXTRA
exec $PY "$HERE/ptb_to_receipt.py" "$BENCH" "$RAW" "$OUT"
