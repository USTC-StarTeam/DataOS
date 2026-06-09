#!/usr/bin/env bash
# 在钉死镜像里跑 run_one.py 的薄封装。DataOS 不感知 docker —— 这层是运行时(你/subagent)的。
#
# 用法(钉一张卡跑一个实验):
#   IMAGE=dataos@sha256:<digest> GPU=6 WORKSPACE="$PWD/workspace" \
#     run-dataos/scripts/run_in_docker.sh --project P --study S --spec X --data /ws/prepared --workspace /ws
#
# 复现一个实验(同镜像 + manifest):
#   IMAGE=dataos@sha256:<digest> GPU=6 WORKSPACE="$PWD/workspace" \
#     run-dataos/scripts/run_in_docker.sh --replay /ws/<study>/experiments/<name>/manifest.json --tol 0
#
# 约定:WORKSPACE 挂到容器内 /ws;给 run_one.py 传 --workspace /ws、--data /ws/...。
set -euo pipefail

IMAGE="${IMAGE:?set IMAGE=<image@digest> (用 digest,不要用 :tag)}"
GPU="${GPU:-0}"
WORKSPACE="${WORKSPACE:-$PWD/workspace}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

exec docker run --rm \
  --gpus "\"device=${GPU}\"" \
  -v "${WORKSPACE}:/ws" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e DATAOS_IMAGE="${IMAGE}" \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  "${IMAGE}" \
  python run-dataos/scripts/run_one.py "$@"
