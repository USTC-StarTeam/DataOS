# 真实验证指南

这份文档说明了两个超出单元测试范围的手工验证入口：

- `scripts/run_all_operator_real_checks.py`
- `scripts/run_live_llm_runtime_demo.py`

不要把 API 密钥写进仓库。需要的凭证只通过 shell 环境变量或本地未跟踪的 `.env` 文件传入。

## 环境变量

如果要调用兼容 OpenAI 的在线模型，请先在 shell 里设置下面这些变量：

- `DASHSCOPE_BASE_URL` 或 `STARTEAM_BASE_URL`
- `DASHSCOPE_API_KEY` 或 `STARTEAM_API_KEY`
- `DASHSCOPE_MODEL` 或 `STARTEAM_MODEL`

`run_all_operator_real_checks.py` 只有在 `quality_filter` 使用 `llm_judge` 模式时才会读取这些变量。  
`run_live_llm_runtime_demo.py` 会用这些变量驱动 `run_task()` 里的规划和复盘阶段。

## 真实算子检查

`scripts/run_all_operator_real_checks.py` 使用真实的 `alpaca_eval` JSON 导出作为输入来源，先从中整理出一份小规模但结构规范的 SFT 数据集，然后按顺序跑内置算子：

- `import_dataset` from `hf_local`
- `import_dataset` from `parquet`
- `schema_guard`
- `pii_redact`
- `dedup`
- `length_filter`
- `lang_filter`
- `quality_filter`
- `category_tag`
- `category_balance`
- `curriculum_sort`
- `instruction_synthesis_template`
- `sample`
- `format_convert`
- `mixture`
- `less`

这里“真实”的意思是算子链路全部真实执行；只有最后的 runtime 尾段是 stub。  
也就是在算子执行完成后，脚本会额外产出一个 stub `ModelArtifact` 和一个 stub `EvalArtifact`，这样可以覆盖完整的集成形态，但不用再支付真实训练和真实评测的成本。

默认的 `less` 模式是 `stub`，因为仓库本身并没有内置 LESS 仓库。如果你要跑真正的 LESS 流程，需要显式传入 `--less-mode less --less-repo-path /path/to/LESS`。

```bash
REPO_ROOT=/path/to/DataOS
cd "$REPO_ROOT"

uv sync --extra dev --extra train --extra eval
source .venv/bin/activate

BASE_MODEL=/absolute/path/to/your/local/Qwen2.5-0.5B-snapshot
ALPACA_EVAL_JSON=/absolute/path/to/alpaca_eval.json

python scripts/run_all_operator_real_checks.py \
  --workspace /tmp/dataos-operator-real-checks \
  --alpaca-eval-json "$ALPACA_EVAL_JSON" \
  --base-model "$BASE_MODEL" \
  --quality-filter-mode llm_judge \
  --less-mode stub \
  --overwrite
```

汇总结果会写到：

```bash
/tmp/dataos-operator-real-checks/operator_real_checks_summary.json
```

## Live LLM Runtime Demo

`scripts/run_live_llm_runtime_demo.py` 的目标更聚焦。它不是测试整套算子库，而是专门验证 `dataos.runtime.run_task()` 里的“大模型推荐算子”流程，输入数据使用真实的 `gsm8k` 子集。

这个脚本会做这些事：

- 从 JSON 读取真实 GSM8K 样本
- 组装一个很小的 `DatasetDict`，并故意放入一条重复训练样本
- 把 runtime 可用算子限制为 `dedup` 和 `sample`
- 使用兼容 OpenAI 的在线模型做规划和复盘
- 把训练和评测都保持在 stub 模式

这个 demo 的重点，是检查 live plan payload、实际执行的算子步骤，以及归档后的实验报告是否正确，而不是去支付真实微调和真实 benchmark 评测的成本。

```bash
REPO_ROOT=/path/to/DataOS
cd "$REPO_ROOT"

uv sync --extra dev --extra train --extra eval
source .venv/bin/activate

GSM8K_JSON=/absolute/path/to/gsm8k_rows.json

python scripts/run_live_llm_runtime_demo.py \
  --workspace /tmp/dataos-live-llm-demo \
  --gsm8k-json "$GSM8K_JSON" \
  --overwrite
```

汇总结果会写到：

```bash
/tmp/dataos-live-llm-demo/live_llm_runtime_demo_summary.json
```
