import json
from pathlib import Path

path = Path(
    "/data2/hcgu/dataos_agentver_runs/ptb-bfcl-4b2/round-0001/"
    "xlam-clean-singlecall-lora/spec.json"
)
spec = json.loads(path.read_text(encoding="utf-8"))
spec["pre_cmds"] = [
    'export TMPDIR="/tmp/job_$SLURM_JOB_ID"',
    'export TORCHINDUCTOR_CACHE_DIR="/data2/hcgu/dataos_agentver_runs/ptb-bfcl-4b2/'
    'round-0001/xlam-clean-singlecall-lora/cache/inductor"',
    'export TRITON_CACHE_DIR="/data2/hcgu/dataos_agentver_runs/ptb-bfcl-4b2/'
    'round-0001/xlam-clean-singlecall-lora/cache/triton"',
    'export TORCH_HOME="/data2/hcgu/dataos_agentver_runs/ptb-bfcl-4b2/round-0001/'
    'xlam-clean-singlecall-lora/cache/torch"',
    'mkdir -p "$TMPDIR/culib" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"',
    'LIBCUDA_PATH=$(ldconfig -p | awk \'/libcuda.so.1/{print $NF; exit}\')',
    'ln -sf "$LIBCUDA_PATH" "$TMPDIR/culib/libcuda.so"',
    'export LIBRARY_PATH="$TMPDIR/culib${LIBRARY_PATH:+:$LIBRARY_PATH}"',
    'export TRITON_LIBCUDA_PATH="$TMPDIR/culib"',
    "python -c 'import time, torch; x=torch.empty(16 * 2**30, dtype=torch.uint8, "
    "device=\"cuda\"); time.sleep(2100)' &",
]
path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
