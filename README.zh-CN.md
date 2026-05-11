# DataOS 中文说明

[English README](README.md)

DataOS 当前收敛为一个面向 SFT 数据工作流的本地优先、单实验 runtime。
LLM 主要负责数据算子的计划和实验结束后的 review；算子执行、训练、评测和
artifact 落盘仍由本地代码主导。对于显式需要打分或数据增强的场景，个别
evaluator / generator 算子可以选择调用 OpenAI-compatible 模型。

当前主文档是 [docs/dataos_overview.md](docs/dataos_overview.md)，算子中文说明见
[docs/operators_zh.md](docs/operators_zh.md)。

## 当前边界

- 数据工件统一使用 Hugging Face `Dataset` / `DatasetDict` 的
  `save_to_disk` 目录。
- `run_task()` 在本地 sandbox 中执行一次实验。
- 数据算子来自本地 registry，执行过程是确定性的。
- `train_llamafactory` 和 `eval_model` 是固定 runtime stage，不是 LLM
  计划出来的 operator。
- 评测层只保留 lm-eval 后端兼容和一个自然语言到 lm-eval task 的轻量 helper。
- 不再维护 research 层、daemon loop、project queue 或异步 agent 控制面。

## 安装

```bash
uv sync --extra dev
```

可选训练 / 评测依赖：

```bash
uv sync --extra train
uv sync --extra eval
```

## 最小入口

```python
from pathlib import Path

from dataos.runtime import run_task

result = run_task(
    task_template=Path("dataos/templates/sft_v1.json"),
    datasets=[Path("/path/to/hf_dataset")],
    workspace_dir=Path("workspace"),
)

print(result.experiment.status)
print(result.review)
```

## CLI

```bash
dataos operators list
dataos train --task dataos/templates/sft_v1.json --dataset tatsu-lab/alpaca,/path/to/hf_dataset
```

## 校验

```bash
uv run ruff check .
```
