# 真实闭环 Demo：AlpacaEval -> 算子处理 -> SFT -> IFEval

本文档对应脚本：

- `scripts/run_real_pipeline_demo.py`

它的目标不是跑一个 stub 或玩具样例，而是在 DataOS 当前固定 runtime 下，真实走通下面这条链路：

1. 读取真实 `tatsu-lab/alpaca_eval` 原始 JSON。
2. 规范化成 DataOS / LLaMA-Factory 可消费的 `instruction / input / response` 数据。
3. 通过 DataOS `import_dataset`、`dedup`、`sample` 算子做真实处理。
4. 用 `LLaMA-Factory` 对本地 Qwen 2.5 0.5B 做真实 LoRA SFT。
5. 用真实 `google/IFEval` 样本做真实推理和真实打分。

## 这个脚本实际做了什么

`scripts/run_real_pipeline_demo.py` 的运行流程如下：

1. 从 `--alpaca-eval-json` 读取真实 `alpaca_eval.json`。
2. 把每条样本转换成：
   - `instruction = row["instruction"]`
   - `input = ""`
   - `response = row["output"]`
3. 将转换后的训练数据保存成 Hugging Face `DatasetDict(train/validation)`。
4. 从 `--ifeval-json` 读取真实 IFEval 行，并保存成 Hugging Face `DatasetDict(train)`。
5. 用 `ImportDatasetOperator` 把 AlpacaEval 训练数据导入成 DataOS `DatasetArtifact`。
6. 用 `DedupOperator(keys=["instruction"])` 做真实去重。
7. 用 `SampleOperator` 对 `train` 和 `validation` 做真实裁剪。
8. 用 `run_train_stage()` 调用 `LLaMA-Factory` 跑真实 LoRA SFT。
9. 用 `run_eval_stage()` 在真实 IFEval 样本上跑 `native + transformers` 评测。
10. 将 summary 写到 `real_demo_summary.json`。

## 输入数据说明

### 训练输入：`tatsu-lab/alpaca_eval`

这里用的不是自造 instruction，也不是仓库里的 stub 数据，而是真实的 `tatsu-lab/alpaca_eval` 原始 JSON。

这个数据集原始字段至少包括：

- `instruction`
- `output`
- `generator`
- `dataset`

由于 DataOS 当前的 `LLaMA-Factory` 导出逻辑要求训练列名里必须有 `instruction` 和 `response`，所以 demo 会在脚本里做一个最小、直接的字段映射：

- `instruction -> instruction`
- `output -> response`
- `input -> ""`

也就是说，这个 demo 的训练输入仍然是 AlpacaEval，只是把它变成了当前训练 runtime 可接受的列格式。

### 评测输入：`google/IFEval`

评测侧使用真实 IFEval 行，字段包括：

- `key`
- `prompt`
- `instruction_id_list`
- `kwargs`

这些字段会直接喂给 `IFEvalTask`，由它构造请求、执行模型生成，并调用 verifier 做真实打分。

## 我这次实际跑通的准备方式

由于我这台服务器当时对 `huggingface.co` 的 DNS 解析不稳定，所以我采用了“联网机器取真实数据文件，服务器负责训练和评测”的方式。这样不影响 demo 的真实性，因为训练输入和评测输入都仍然来自真实数据源。

我实际准备了两个文件：

- `alpaca_eval.json`
- `ifeval_rows.json`

其中：

- `alpaca_eval.json` 是从 `tatsu-lab/alpaca_eval` 下载的真实原始 JSON
- `ifeval_rows.json` 是从 `google/IFEval` 取出的真实行子集

## 我实际跑通的命令

### 1. 在一台能访问 Hugging Face 的机器上准备真实数据文件

```bash
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

(out / 'alpaca_eval.json').write_text(
    json.dumps(alpaca, ensure_ascii=False),
    encoding='utf-8',
)
(out / 'ifeval_rows.json').write_text(
    json.dumps(ifeval_rows, ensure_ascii=False, indent=2),
    encoding='utf-8',
)

print(out)
print('alpaca_rows =', len(alpaca))
print('ifeval_rows =', len(ifeval_rows))
PY
```

### 2. 把这两个真实数据文件传到服务器

```bash
scp /tmp/dataos-real-demo-inputs/alpaca_eval.json \
  /tmp/dataos-real-demo-inputs/ifeval_rows.json \
  user@your-server:/tmp/
```

### 3. 在服务器上跑完整 demo

```bash
REPO_ROOT=/path/to/DataOS
cd "$REPO_ROOT"

uv sync --extra dev --extra train --extra eval
source .venv/bin/activate
uv pip install --python .venv/bin/python --upgrade \
  'transformers>=4.51.0,<=4.57.1' \
  'peft>=0.14.0,<=0.17.1'

BASE_MODEL=/absolute/path/to/your/local/Qwen2.5-0.5B-snapshot

python scripts/run_real_pipeline_demo.py \
  --workspace /tmp/dataos-real-alpaca-ifeval \
  --base-model "$BASE_MODEL" \
  --alpaca-eval-json /tmp/alpaca_eval.json \
  --ifeval-json /tmp/ifeval_rows.json \
  --max-train-samples 8 \
  --max-validation-samples 4 \
  --max-eval-samples 4 \
  --overwrite
```

## 这次实跑的关键验证点

这套命令在服务器上真实完成了下面这些动作：

- 真实读取 `tatsu-lab/alpaca_eval` 文件
- 真实生成 Hugging Face `DatasetDict`
- 真实经过 `import_dataset`、`dedup`、`sample`
- 真实调用 `llamafactory-cli train`
- 真实产出 LoRA adapter
- 真实读取 IFEval 样本
- 真实调用模型生成
- 真实执行 IFEval verifier 打分
- 真实写出 `EvalArtifact` 和样本 artifact

## 我这次实跑得到的结果

这次实跑在服务器上已经成功完成，关键结果如下：

- `alpaca_eval_row_count = 805`
- 导入后数据量：`train = 724`，`validation = 81`
- `dedup_removed_counts = {"train": 0, "validation": 0}`
- 采样后数据量：`train = 8`，`validation = 4`
- IFEval 样本数：`4`
- `prompt_strict_accuracy = 0.25`
- `instruction_strict_accuracy = 0.14285714285714285`
- summary 文件：`/tmp/dataos-real-alpaca-ifeval/real_demo_summary.json`

这说明这次不是只把训练命令触发了一下，而是真的把：

- 训练输入准备
- 算子处理
- LoRA adapter 产出
- IFEval 生成
- IFEval verifier 打分

整条链路都完整跑通了。

## 你可以重点看哪些输出

demo 跑完后，最值得检查的是：

- `real_demo_summary.json`
- `sandbox/logs/train_llamafactory.log`
- `sandbox/artifacts/models/...`
- `sandbox/artifacts/evals/...`
- `sandbox/artifacts/results/...`

其中 `real_demo_summary.json` 会集中汇总：

- 导入后数据量
- 去重后数据量
- 采样后数据量
- 模型 artifact 路径
- 训练日志路径
- IFEval 的主指标和明细 summary

## 适用场景

如果你的目标是：

- 验证 DataOS 不是只会跑 stub
- 验证“数据处理算子 + 训练 + 评测”整条链路是否真的闭环
- 给新接手仓库的人提供一条最短真实主线路径

那么这个 demo 比 `scripts/run_ifeval_baseline.py` 更合适。

如果你只是想快速验证“评测栈能不能跑”，那么 `scripts/run_ifeval_baseline.py` 依然更轻、更稳。
