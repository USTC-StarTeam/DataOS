#!/bin/bash
#SBATCH --job-name=xlam-inspect
#SBATCH --nodes=1
#SBATCH --time=00:30:00
set -euo pipefail
RUN_ROOT=/data2/hcgu/dataos_agentver_runs/ptb-bfcl-4b2/round-0001/xlam-clean-singlecall-lora
POOL=/data2/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v2/xlam_function_calling_60k
AGENTVER=/data/hcgu/users/tjh/dataos_agentver
export DATAOS_OPERATOR_PLUGINS="$AGENTVER/resources/custom_operators:$RUN_ROOT/custom_operators"
export TMPDIR="/tmp/job_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
source "$AGENTVER/configs/environments/PTB_H100_slurm/envs/dataos_sft/.venv/bin/activate"
python - <<'PY'
import json
from datasets import load_from_disk
import dataos_operator as o

pool = "/data2/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v2/xlam_function_calling_60k"
ds = load_from_disk(pool)
print("TYPE", type(ds), "SPLITS", list(ds), flush=True)
train = ds["train"]
print("ROWS", len(train), "FEATURES", train.features, flush=True)
for i in range(3):
    print("ROW", i, json.dumps(train[i], ensure_ascii=False)[:12000], flush=True)
for name in ("function_call_normalizer", "function_call_filter", "deduplicate_selector", "random_sample_selector"):
    print("USAGE", name, flush=True)
    print(o.usage(name), flush=True)
PY
