# DataOS 中文说明

[English README](README.md)

## 项目概括

DataOS 是一个面向监督微调（SFT）场景的本地优先数据系统。它把数据集、算子、训练、评测和实验产物统一放进一个固定 runtime 里执行：

- 数据工件统一使用 Hugging Face `Dataset` / `DatasetDict` 的 `save_to_disk` 目录。
- 数据处理算子在受控 sandbox 中运行，并产出带元数据的 typed artifact。
- 训练和评测不是让 LLM 随意编排，而是固定 runtime stage。
- LLM 当前主要负责 plan / review，不直接执行训练与评测。

换句话说，这个仓库的重点不是“聊天式 agent”，而是“可落盘、可追踪、可重复执行”的数据与训练实验 runtime。

## 仓库结构

- `dataos/core`: Artifact、TaskTemplate、Sandbox、Experiment、Registry 等核心抽象。
- `dataos/operators`: 内置数据算子，目前包括 `import_dataset`、`dedup`、`sample`、`format_convert`、`less`。
- `dataos/train`: 真实训练阶段，当前接入 `LLaMA-Factory`。
- `dataos/eval`: 真实评测阶段，支持 `native` provider、`transformers` / `openai_compatible` adapter，以及多类 task 定义。
- `dataos/runtime`: 组合 task、artifact、agent、train、eval 的统一运行入口。
- `dataos/research`: 面向研究任务编排的 project / task / worker / scheduler / agent 层。
- `dataos/templates`: 任务模板示例。
- `scripts`: 可执行脚本；本次补充了一个真实可复现的 runtime demo。
- `tests`: 单元测试和集成测试。

第一批稳定算子的说明文档见 `docs/stable_operator_library_v1.md`。

## 当前需要注意的真实情况

仓库自带的 `dataos/templates/sft_v1.json` 默认还是：

- `train_config.mode = "stub"`
- `eval_config.provider = "stub"`

所以如果你直接照英文 README 里的默认模板运行，它不是一个“真实训练 + 真实评测”的 demo。

本中文 README 下面提供的是一套我已经在服务器本地实际跑通的、**非 stub**、**真实训练**、**真实评测** 的 demo 命令。

## 已验证的真实 demo

### 这套 demo 覆盖了什么

`scripts/run_real_pipeline_demo.py` 会按顺序执行：

1. 构造一个小型 Hugging Face `DatasetDict`。
2. 运行真实 `dedup` 算子。
3. 运行真实 `sample` 算子。
4. 运行真实 `LLaMA-Factory` LoRA 训练。
5. 运行真实 `native + transformers` 评测。
6. 输出真实 artifact、训练日志和评测结果。

它没有走 `stub` 训练，也没有走 `stub` 评测。

### 我实际验证通过的环境

- Ubuntu 24.04
- Python 3.11（由 `uv` 创建 `.venv`）
- NVIDIA RTX 5090
- CUDA 12.8 驱动
- 本地缓存模型：`Qwen/Qwen2.5-0.5B`

### 我实际跑通的命令

下面这组命令是我在本机真实跑通的命令。它会生成一个完整 workspace，并在其中留下：输入数据、去重后数据、采样后数据、LoRA adapter、训练日志、评测 artifact 和 summary。

我验证时使用的是本地 Hugging Face 缓存中的 `Qwen/Qwen2.5-0.5B` snapshot；下面改写成通用占位路径，避免把机器上的绝对路径写进仓库。

```bash
REPO_ROOT=/path/to/DataOS
cd "$REPO_ROOT"

uv sync --extra dev --extra train --extra eval
source .venv/bin/activate
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  --upgrade --force-reinstall torch torchvision torchaudio

BASE_MODEL=/absolute/path/to/your/local/Qwen2.5-0.5B-snapshot

python scripts/run_real_pipeline_demo.py \
  --workspace /tmp/dataos-real-demo \
  --base-model "$BASE_MODEL" \
  --overwrite
```

### 我本地实跑得到的结果

这次实跑成功完成了真实训练和真实评测，关键结果包括：

- 训练前 `train` split 有 5 条样本，其中 1 条重复。
- `dedup` 后 `train` split 变为 4 条。
- `sample` 后仍保留 `train=4`、`test=2`。
- `train_llamafactory` 产出了真实 LoRA adapter 目录。
- `eval_model` 在 `test` split 上完成了真实推理和打分。
- 这次 demo 的 `gsm8k exact_match` 结果为 `1.0`（2/2）。

输出 summary 会写到：

```bash
/tmp/dataos-real-demo/real_demo_summary.json
```

训练日志会写到 summary 里的 `train_log` 对应路径。

## 关于 `run_task()` 和 agent 版完整闭环

仓库的 `run_task()` 还会包含：

- `LLMPlanningAgent.plan()`
- `LLMPlanningAgent.review()`

也就是说，如果你想跑“LLM 自动规划数据算子 + 固定 runtime 训练评测”的完整 agent 闭环，还需要一个可用的 JSON LLM 客户端，例如：

- 仓库当前默认支持的 OpenAI-compatible endpoint
- 或你自己传入的自定义 `LLMClient`

本 README 里给出的已验证 demo，故意聚焦在**本仓库当前最稳定、最容易本地复现、并且不依赖私有 API 凭证** 的真实 runtime 主链路上。

## 建议阅读顺序

如果你第一次接手这个仓库，建议按这个顺序看：

1. `README.md`
2. `README.zh-CN.md`
3. `dataos/runtime/runner.py`
4. `dataos/train/llamafactory.py`
5. `dataos/eval/runner.py`
6. `dataos/operators/`
7. `dataos/research/`

这样会比较容易先抓住 runtime 主线，再理解 agent 和 research orchestration 层。
