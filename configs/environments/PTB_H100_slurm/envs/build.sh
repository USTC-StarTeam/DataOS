#!/bin/bash
# Build one uv env: ./build.sh {sft|rl|eval}. Reproducible: same pins + cuda index → same env.
# torch is pinned with its +cuXXX LOCAL version (only present on the CUDA wheel index) and installed
# IN THE SAME resolution as the rest — so a loose dep (e.g. llamafactory's torch>=X) can't upgrade it.
# flash-attn is best-effort (needs nvcc + matching torch).
set -euo pipefail
E="${1:?usage: build.sh sft|rl|eval}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
# FA_CONDA: reuse the exact-matching flash_attn from the source conda env. There is no
# prebuilt flash-attn wheel for these torch combos (2.7.4.post1 predates torch2.8) and the
# system nvcc (13.2) mismatches torch's CUDA (12.9/12.4) so compiling fails — but each uv env
# is pinned bit-for-bit to a conda env (py/torch/cuda/cxx11abi), so that conda env's compiled
# flash_attn_2_cuda.so drops in and loads.
case "$E" in
  eval) PY=3.12; TORCH="torch==2.9.0+cu128"; IDX="https://download.pytorch.org/whl/cu128"; FA_CONDA="" ;;
  sft)  PY=3.12; TORCH="torch==2.8.0+cu129"; IDX="https://download.pytorch.org/whl/cu129"; FA_CONDA="/data/hcgu/.conda/envs/Llama-op020/lib/python3.12/site-packages" ;;
  rl)   PY=3.11; TORCH="torch==2.6.0+cu124"; IDX="https://download.pytorch.org/whl/cu124"; FA_CONDA="/data/hcgu/.conda/envs/dataos-verl/lib/python3.11/site-packages" ;;
  *) echo "unknown env $E"; exit 1 ;;
esac
V="$ROOT/$E/.venv"
echo ">>> [$E] uv venv --python $PY"
rm -rf "$V"
"$UV" venv --python "$PY" "$V"
echo ">>> [$E] install torch ($TORCH, pinned) + requirements in one resolution (extra-index $IDX)"
"$UV" pip install --python "$V/bin/python" "$TORCH" -r "$ROOT/$E/requirements.txt" \
  --extra-index-url "$IDX" --index-strategy unsafe-best-match
echo ">>> [$E] torch check"
"$V/bin/python" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
if [ -n "$FA_CONDA" ] && [ -d "$FA_CONDA" ]; then
  echo ">>> [$E] flash-attn: reuse exact-matching build from $FA_CONDA"
  SP="$V/lib/python$PY/site-packages"
  if cp -r "$FA_CONDA"/flash_attn "$FA_CONDA"/flash_attn-*.dist-info "$FA_CONDA"/flash_attn_2_cuda*.so "$SP"/ 2>/dev/null; then
    "$V/bin/python" -c "import flash_attn;print('    flash_attn',flash_attn.__version__,'OK')" \
      || echo "!!! [$E] flash-attn copied but import failed — env still usable (sdpa)"
  else
    echo "!!! [$E] flash-attn reuse failed (source missing) — env usable without it (sdpa)"
  fi
fi
echo ">>> [$E] done: $V"
