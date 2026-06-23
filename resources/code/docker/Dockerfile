# DataOS 训练/评估镜像 —— 复现的「环境」层。
#
# 分工:uv 在镜像里装并锁死 python 层(uv.lock);docker 把「OS + CUDA + python + uv 装出来的包」
# 整层冻成一个 image。**按 digest 钉死**:build 后 `docker images --digests` 拿 sha256,运行/记录都用
# `image@sha256:...`(不要用 :tag,tag 会动)。DataOS 代码不感知 docker —— 这层属于运行时/skill。
#
# ⚠️ 基础镜像也要钉 digest:下面的 :tag 仅占位,生产请换成 <base>@sha256:...。
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# git:dataos-operator 是 git 依赖,且 provenance 要采集 git sha
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# uv:确定性安装 python 层
RUN pip install --no-cache-dir uv

WORKDIR /opt/dataos

# 1) 先只 COPY 锁文件 + 装依赖(不装本项目)—— 让重依赖层可缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra train --extra eval --no-install-project

# 2) 再 COPY 源码 + 运行资源,装本项目(快)
COPY src ./src
COPY resources ./resources
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --extra train --extra eval

# uv 装进 /opt/dataos/.venv;放进 PATH(llamafactory-cli 等),HF 走镜像源
ENV PATH="/opt/dataos/.venv/bin:${PATH}"
ENV HF_ENDPOINT="https://hf-mirror.com"

# 无默认 CMD。运行示例(钉卡 + 挂 workspace/HF cache + 注入自身 digest 供 manifest 记录):
#   docker run --rm --gpus '"device=0"' \
#     -v "$PWD/workspace:/ws" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
#     -e DATAOS_IMAGE=dataos@sha256:<digest> dataos@sha256:<digest> \
#     python scripts/run_one.py --project P --study S --spec X --data /ws/prepared --workspace /ws
# 复现:把上面的命令换成 `... scripts/run_one.py --replay /ws/<study>/experiments/<name>/manifest.json`
