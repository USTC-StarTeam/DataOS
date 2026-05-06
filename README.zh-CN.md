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
- `dataos/train`: 训练阶段，当前接入 `LLaMA-Factory`。
- `dataos/eval`: 评测阶段，支持 `native` provider、`transformers` / `openai_compatible` adapter，以及多类 task 定义。
- `dataos/runtime`: 组合 task、artifact、agent、train、eval 的统一运行入口。
- `dataos/research`: 面向研究任务编排的 project / task / worker / scheduler / agent 层。
- `dataos/templates`: 任务模板与导入 spec。
- `scripts`: 实际运行脚本。
- `tests`: 单元测试和集成测试。

第一批稳定算子的说明文档见 `docs/stable_operator_library_v1.md`。

## 当前推荐的 demo

从现在开始，中文 README 里默认推荐的 demo 脚本统一是：

- `scripts/run_ifeval_baseline.py`

这个脚本的定位很明确：

- 它不改模型代码，直接对一个本地模型跑 IFEval baseline。
- 它的输入是一个 Hugging Face `save_to_disk` 格式的 IFEval 数据目录。
- 它的输出是 DataOS 标准的 `EvalArtifact` 和样本结果 artifact。
- 它不依赖 `run_task()` 的 plan/review LLM，也不需要你先把完整 research orchestration 跑起来。

仓库里和 “Alpaca 训练 + IFEval 评测” 对应的任务模板是 `dataos/templates/alpaca_ifeval_v1.json`。那个模板描述的是同一个主题方向：训练侧用 Alpaca 风格数据，评测侧用 IFEval。当前这份 README 选择把 `scripts/run_ifeval_baseline.py` 作为默认 demo，是因为它更稳定、更直接，也更容易先把评测链路跑通。

## 已验证的运行方式

### 这套 demo 覆盖了什么

`scripts/run_ifeval_baseline.py` 会完成下面这条真实链路：

1. 读取一个 IFEval 格式的 Hugging Face `Dataset` / `DatasetDict` 目录。
2. 构造一个指向本地基础模型的 `ModelArtifact`。
3. 调用 `run_eval_stage()`。
4. 通过 `native` provider 和 `transformers` adapter 执行真实推理。
5. 产出真实 `EvalArtifact` 和样本结果 artifact。

它是一个真实评测 demo，不是 stub 评测。

### 我实际验证通过的环境

- Ubuntu 24.04
- Python 3.11（由 `uv` 创建 `.venv`）
- NVIDIA RTX 5090
- CUDA 12.8 驱动
- 本地缓存模型：`Qwen/Qwen2.5-0.5B`

额外说明：我在这台机器上实际跑这个脚本时，把环境里的 `transformers` 升级到了 `4.57.x`，因为较旧版本缺少 `Qwen2Tokenizer`，会导致本地 Qwen 2.5 快照无法被脚本加载。

### 我实际跑通的命令

下面这组命令是我在服务器上真实跑通的命令。为了让命令不依赖私有数据路径，我先用一个极小的 IFEval 格式样例数据集生成了本地 `save_to_disk` 目录，再把它喂给 `scripts/run_ifeval_baseline.py`。

```bash
REPO_ROOT=/path/to/DataOS
cd "$REPO_ROOT"

uv sync --extra dev --extra train --extra eval
source .venv/bin/activate
uv pip install --python .venv/bin/python --upgrade 'transformers>=4.45,<5'

python - <<'PY'
from pathlib import Path
from datasets import Dataset, DatasetDict
import shutil

rows = {
    "key": ["sample-1", "sample-2"],
    "prompt": [
        "Write a one-paragraph note ending with the phrase Have a great day.",
        "Reply in lowercase only and avoid the word secret.",
    ],
    "instruction_id_list": [
        ["startend:end_checker", "length_constraints:number_words"],
        ["change_case:english_lowercase", "keywords:forbidden_words"],
    ],
    "kwargs": [
        [
            {"end_phrase": "Have a great day."},
            {"relation": "at least", "num_words": 5},
        ],
        [
            {},
            {"forbidden_words": ["secret"]},
        ],
    ],
}

path = Path('/tmp/dataos-ifeval-mini')
if path.exists():
    shutil.rmtree(path)
DatasetDict({"train": Dataset.from_dict(rows)}).save_to_disk(str(path))
print(path)
PY

BASE_MODEL=/absolute/path/to/your/local/Qwen2.5-0.5B-snapshot

python scripts/run_ifeval_baseline.py \
  --workspace /tmp/dataos-ifeval-baseline \
  --dataset /tmp/dataos-ifeval-mini \
  --model "$BASE_MODEL" \
  --max-samples 2 \
  --batch-size 1 \
  --max-new-tokens 128
```

### 我这次实跑得到的结果

这次实跑已经成功完成，脚本正常输出了真实 `EvalArtifact`。关键结果是：

- 评测任务：`ifeval`
- 评测 provider：`native`
- adapter：`transformers`
- 样本数：`2`
- `prompt_strict_accuracy = 0.0`
- `instruction_strict_accuracy = 0.25`
- 输出目录：`/tmp/dataos-ifeval-baseline/sandbox/artifacts/evals/...`

这里的分数高低不重要，重要的是：脚本本身、模型加载、IFEval 请求构建、真实推理、打分和 artifact 落盘这条链路已经跑通了。

## 其他 demo 的定位

仓库里现有的其他 demo / 校验脚本可以继续保留做内部验证，但中文 README 后续默认只把 `scripts/run_ifeval_baseline.py` 当作对外 demo 入口。

如果你要看更完整的 task template 背景，可以再读：

- `dataos/templates/alpaca_ifeval_v1.json`

如果你要理解 `run_task()` 的固定 runtime 主线，可以再读：

- `dataos/runtime/runner.py`
- `dataos/train/llamafactory.py`
- `dataos/eval/runner.py`

## 建议阅读顺序

如果你第一次接手这个仓库，建议按这个顺序看：

1. `README.md`
2. `README.zh-CN.md`
3. `scripts/run_ifeval_baseline.py`
4. `dataos/templates/alpaca_ifeval_v1.json`
5. `dataos/runtime/runner.py`
6. `dataos/train/llamafactory.py`
7. `dataos/eval/runner.py`
8. `dataos/research/`

这样会比较容易先抓住当前默认 demo，再往下理解 runtime 和 research orchestration。
