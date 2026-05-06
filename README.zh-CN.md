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

- `scripts/run_real_pipeline_demo.py`

这个脚本的定位很明确：

- 它走的是完整的真实闭环，不是单独的评测旁路。
- 它的训练输入来自真实 `tatsu-lab/alpaca_eval`。
- 它的中间会经过真实 `import_dataset`、`dedup`、`sample` 算子。
- 它会调用真实 `LLaMA-Factory` 跑 LoRA SFT。
- 它最终会在真实 `google/IFEval` 样本上做评测，并产出 DataOS 标准 artifact。

仓库里和 “Alpaca 训练 + IFEval 评测” 对应的任务模板是 `dataos/templates/alpaca_ifeval_v1.json`。`scripts/run_real_pipeline_demo.py` 现在就是沿着这个方向整理出来的默认入口：它更适合展示 DataOS 当前固定 runtime 的完整主链路，而不只是单独验证评测栈。

如果你只想快速验证 IFEval 评测栈，而不想先跑训练，可以再看：

- `scripts/run_ifeval_baseline.py`

## 已验证的运行方式

### 默认 demo 覆盖了什么

`scripts/run_real_pipeline_demo.py` 会完成下面这条真实链路：

1. 读取真实 `tatsu-lab/alpaca_eval` 原始 JSON。
2. 规范化成可训练的 Hugging Face `DatasetDict(train/validation)`。
3. 通过 `import_dataset`、`dedup`、`sample` 做真实处理。
4. 调用 `run_train_stage()` 和 `LLaMA-Factory` 跑真实 LoRA SFT。
5. 读取真实 IFEval 行并构造成评测数据集。
6. 通过 `native` provider 和 `transformers` adapter 执行真实推理与真实打分。
7. 产出真实模型 artifact、样本 artifact、`EvalArtifact` 和 `real_demo_summary.json`。

它不是 stub，也不是只跑评测的旁路 demo，而是完整的训练加评测闭环。

### 我实际验证通过的环境

- Ubuntu 24.04
- Python 3.11（由 `uv` 创建 `.venv`）
- NVIDIA RTX 5090
- CUDA 12.8 驱动
- 本地缓存模型：`Qwen/Qwen2.5-0.5B`

额外说明：我在这台机器上实际跑这个脚本时，把环境里的 `transformers` 固定到了 `4.57.1`。较旧版本缺少 `Qwen2Tokenizer`，而更高版本又可能和当前 `LLaMA-Factory` 训练链路不兼容，所以 `4.57.1` 是当前这套 demo 更稳妥的交集版本。

### 我实际跑通的命令

下面这组命令是我在服务器上真实跑通的命令。为了让命令不依赖私有路径，我把需要的真实源数据都写成了通用文件输入形式。

```bash
REPO_ROOT=/path/to/DataOS
cd "$REPO_ROOT"

uv sync --extra dev --extra train --extra eval
source .venv/bin/activate
uv pip install --python .venv/bin/python --upgrade 'transformers>=4.51.0,<=4.57.1'
uv pip install --python .venv/bin/python --upgrade 'peft>=0.14.0,<=0.17.1'

BASE_MODEL=/absolute/path/to/your/local/Qwen2.5-0.5B-snapshot

python3 - <<'PY'
import json
import pathlib
import urllib.request

out = pathlib.Path('/tmp/dataos-real-demo-inputs')
out.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(
    'https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json?download=true',
    timeout=60,
) as resp:
    alpaca = json.load(resp)

with urllib.request.urlopen(
    'https://datasets-server.huggingface.co/rows?dataset=google%2FIFEval&config=default&split=train&offset=0&length=8',
    timeout=30,
) as resp:
    ifeval_rows = [item['row'] for item in json.load(resp)['rows']]

(out / 'alpaca_eval.json').write_text(json.dumps(alpaca, ensure_ascii=False), encoding='utf-8')
(out / 'ifeval_rows.json').write_text(json.dumps(ifeval_rows, ensure_ascii=False, indent=2), encoding='utf-8')
print(out)
PY

python scripts/run_real_pipeline_demo.py \
  --workspace /tmp/dataos-real-alpaca-ifeval \
  --base-model "$BASE_MODEL" \
  --alpaca-eval-json /tmp/dataos-real-demo-inputs/alpaca_eval.json \
  --ifeval-json /tmp/dataos-real-demo-inputs/ifeval_rows.json \
  --max-train-samples 8 \
  --max-validation-samples 4 \
  --max-eval-samples 4 \
  --overwrite
```

### 我这次实跑得到的结果

这次实跑已经成功完成，脚本正常输出了真实模型 artifact 和真实 `EvalArtifact`。关键结果是：

- `alpaca_eval_row_count = 805`
- 导入后数据量：`train = 724`，`validation = 81`
- 采样后数据量：`train = 8`，`validation = 4`
- 评测任务：`ifeval`
- 评测 provider：`native`
- adapter：`transformers`
- 样本数：`4`
- `prompt_strict_accuracy = 0.25`
- `instruction_strict_accuracy = 0.14285714285714285`
- summary：`/tmp/dataos-real-alpaca-ifeval/real_demo_summary.json`

这里的分数高低不重要，重要的是：真实输入、真实算子、真实训练、真实推理、真实打分和 artifact 落盘这条链路已经跑通了。

## 其他 demo 的定位

仓库里现有的其他 demo / 校验脚本可以继续保留做内部验证，但中文 README 后续默认把 `scripts/run_real_pipeline_demo.py` 当作对外 demo 入口。

其中 `scripts/run_ifeval_baseline.py` 现在更适合作为轻量评测 demo：

- 它不跑训练
- 它直接评测本地基础模型
- 它更适合单独验证 IFEval 评测链路是否正常

如果你要看默认完整链路的拆解说明，可以直接读：

- `docs/real_demo_alpaca_ifeval.md`

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
3. `scripts/run_real_pipeline_demo.py`
4. `docs/real_demo_alpaca_ifeval.md`
5. `dataos/templates/alpaca_ifeval_v1.json`
6. `dataos/runtime/runner.py`
7. `dataos/train/llamafactory.py`
8. `dataos/eval/runner.py`
9. `dataos/research/`

这样会比较容易先抓住当前默认完整 demo，再往下理解 runtime 和 research orchestration。
