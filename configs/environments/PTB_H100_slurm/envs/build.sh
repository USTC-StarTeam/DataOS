#!/bin/bash
# Build one uv env: ./build.sh {sft|rl|eval}. Reproducible: same pins + cuda index → same env.
# torch is pinned with its +cuXXX LOCAL version (only present on the CUDA wheel index) and installed
# IN THE SAME resolution as the rest — so a loose dep (e.g. llamafactory's torch>=X) can't upgrade it.
# flash-attn is best-effort (needs nvcc + matching torch).
set -euo pipefail
E="${1:?usage: build.sh sft|rl|eval}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
# FA_CONDA: reuse the exact-matching flash_attn from the source conda env. WHERE that env
# lives is a fact about the machine, so it comes in as FA_CONDA_SFT / FA_CONDA_RL rather than
# being written here — unset simply means "no donor", which the copy step below already handles
# (the env stays usable on sdpa). There is no
# prebuilt flash-attn wheel for these torch combos (2.7.4.post1 predates torch2.8) and the
# system nvcc (13.2) mismatches torch's CUDA (12.9/12.4) so compiling fails — but each uv env
# is pinned bit-for-bit to a conda env (py/torch/cuda/cxx11abi), so that conda env's compiled
# flash_attn_2_cuda.so drops in and loads.
case "$E" in
  eval) PY=3.12; TORCH="torch==2.9.0+cu128"; IDX="https://download.pytorch.org/whl/cu128"; FA_CONDA="" ;;
  sft)  PY=3.12; TORCH="torch==2.8.0+cu129"; IDX="https://download.pytorch.org/whl/cu129"; FA_CONDA="${FA_CONDA_SFT:-}" ;;
  dataos_sft) PY=3.12; TORCH="torch==2.8.0+cu129"; IDX="https://download.pytorch.org/whl/cu129"; FA_CONDA="${FA_CONDA_SFT:-}" ;;  # sft 基线 + repo 工具(tools/* 本地装,版本= env.yaml dataos_tools 声明)
  rl)   PY=3.11; TORCH="torch==2.6.0+cu124"; IDX="https://download.pytorch.org/whl/cu124"; FA_CONDA="${FA_CONDA_RL:-}" ;;
  train-cli) PY=3.12; TORCH=""; IDX=""; FA_CONDA="" ;;  # thin orchestrator env: pydantic + dataos-train only
  *) echo "unknown env $E"; exit 1 ;;
esac
REPO="$(cd "$ROOT/../../../.." && pwd)"   # deploy checkout root (tools/ lives there)
V="$ROOT/$E/.venv"
echo ">>> [$E] uv venv --python $PY"
rm -rf "$V"
"$UV" venv --python "$PY" "$V"
if [ -n "$TORCH" ]; then
  echo ">>> [$E] install torch ($TORCH, pinned) + requirements in one resolution (extra-index $IDX)"
  "$UV" pip install --python "$V/bin/python" "$TORCH" -r "$ROOT/$E/requirements.txt" \
    --extra-index-url "$IDX" --index-strategy unsafe-best-match
  echo ">>> [$E] torch check"
  "$V/bin/python" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
else
  echo ">>> [$E] install requirements (no torch — thin env)"
  "$UV" pip install --python "$V/bin/python" -r "$ROOT/$E/requirements.txt"
fi
# repo 自带工具从本地 checkout 安装——版本即 env.yaml 的 dataos_tools 声明(有测试兜底)。
# ★ 一个发行包 dataos-tools 就带齐三个 CLI(dataos-data/-train/-eval)。main 上曾漏装 eval:
# env.yaml 的 provides 和 dataos_tools 都声明了它、task.yaml 的评测命令也调它,唯独这里没装,
# 而老 venv 建于拆包之前自带该命令,于是分歧从未被任何一次运行暴露。四包并成一包之后,
# "声明了却没装"在这里不再可能 —— 装的就是声明的那一个东西。
case "$E" in
  dataos_sft)
    echo ">>> [$E] install repo tools from $REPO/src"
    "$UV" pip install --python "$V/bin/python" "$REPO/src/dataos-tools" "$REPO/src/operator"
    ;;
  train-cli)
    echo ">>> [$E] install repo tools from $REPO/src (dataos-tools carries the train CLI)"
    "$UV" pip install --python "$V/bin/python" "$REPO/src/dataos-tools"
    ;;
esac
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
