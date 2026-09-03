#!/bin/bash
# Build the isolated DataOS/Curation-Bench runtimes from their pinned lock files.
set -euo pipefail

TARGET="${1:?usage: build.sh dataos_cli|harness|train|eval|all}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/../../../.." && pwd)"
CURATION_ROOT="${CURATION_ROOT:?set CURATION_ROOT to the pinned curation-bench checkout}"
UV="${UV:-$HOME/.local/bin/uv}"

build_dataos_cli() {
  local venv="$ROOT/dataos_cli/.venv"
  rm -rf "$venv"
  "$UV" venv --python 3.12 "$venv"
  "$UV" pip install --python "$venv/bin/python" "$REPO/src/dataos-tools" "datasets>=4.6.0"
  "$venv/bin/python" -c \
    "import importlib.metadata as m; print('dataos-tools', m.version('dataos-tools'))"
}

build_harness() {
  "$UV" sync --frozen --project "$CURATION_ROOT"
  "$CURATION_ROOT/.venv/bin/datacuration-bench" --help >/dev/null
}

copy_flash_attn() {
  local venv="$1"
  local donor="${FLASH_ATTN_DONOR:-}"
  if [ -z "$donor" ]; then
    echo "FLASH_ATTN_DONOR is required for the pinned H100 build" >&2
    return 2
  fi
  "$donor/bin/python" - <<'PY'
import flash_attn
import sys
import torch

expected = ("3.11", "2.6.0+cu124", "2.7.4.post1")
actual = (f"{sys.version_info.major}.{sys.version_info.minor}", torch.__version__, flash_attn.__version__)
if actual != expected:
    raise SystemExit(f"flash-attn donor mismatch: expected {expected}, got {actual}")
PY
  local site="$venv/lib/python3.11/site-packages"
  local donor_site="$donor/lib/python3.11/site-packages"
  cp -a "$donor_site/flash_attn" "$donor_site"/flash_attn-*.dist-info \
    "$donor_site"/flash_attn_2_cuda*.so "$site/"
  "$venv/bin/python" -c \
    "import flash_attn, torch; print('train', torch.__version__, flash_attn.__version__)"
}

build_train() {
  local project="$CURATION_ROOT/vendor/curation-train"
  local venv="$project/.venv"
  rm -rf "$venv"
  UV_PROJECT_ENVIRONMENT="$venv" "$UV" sync --frozen --project "$project" \
    --python 3.11 --no-install-package flash-attn
  copy_flash_attn "$venv"
  "$venv/bin/python" -c "import curation_train"
}

build_eval() {
  local project="$CURATION_ROOT/vendor/VLMEvalKit"
  local venv="$project/.venv"
  # The pinned Curation-Bench snapshot's run.py imports this upstream VLMEvalKit module but
  # omitted it from vendor/. Install the exact compatible upstream blob before import/eval.
  install -m 0644 "$ROOT/../compat/result_transfer.py" \
    "$project/vlmeval/utils/result_transfer.py"
  "$UV" run --no-project --python 3.12 "$ROOT/../compat/patch_vlmeval_secret_log.py" \
    --project "$project"
  rm -rf "$venv"
  UV_PROJECT_ENVIRONMENT="$venv" "$UV" sync --frozen --project "$project" --python 3.12
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$venv/bin/python" - <<'PY'
import torch
import vllm
from vlmeval.utils import result_transfer  # noqa: F401

print("eval", torch.__version__, vllm.__version__)
PY
}

case "$TARGET" in
  dataos_cli) build_dataos_cli ;;
  harness) build_harness ;;
  train) build_train ;;
  eval) build_eval ;;
  all)
    build_dataos_cli
    build_harness
    build_train
    build_eval
    ;;
  *) echo "unknown environment $TARGET" >&2; exit 2 ;;
esac
