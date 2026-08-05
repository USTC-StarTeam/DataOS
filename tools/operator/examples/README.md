# 示例

这里存放两类 demo：

- 新的 `policy_*` demo：展示统一的 `PolicyRuntime`，覆盖离线、在线、混合三种执行形态。
- 旧的 evaluator/mapper 流水线 demo：保留作参考，其中一部分需要本地模型快照或额外配置。

所有新的 `policy_*` demo 都直接使用仓库里的两份实验数据：

- `dataset/gsm8k/gsm8k.jsonl`
- `dataset/alpaca/alpaca.jsonl`

> 这两个输入文件位于被 `.gitignore` 忽略的 `dataset/` 目录，不随仓库分发；运行前需自行放置，或把脚本里的输入路径改成仓库自带的小样例数据（如 `tests/fixtures/gsm8k_target_100.jsonl`）。

输出目录统一写到：

- `examples/outputs/`

## 推荐先跑

```bash
PYTHONPATH=. python3 examples/policy_selectors_demo.py
```

覆盖：

- `Loss`
- `Delta Loss`
- `LESS`
- `NICE`
- `NEAR`
- `TSDS`

这个 demo 会同时读取 `gsm8k` 和 `alpaca`，并把各算法选出来的样本写到 `examples/outputs/policy_selectors_demo/`。

```bash
PYTHONPATH=. python3 examples/policy_mixers_and_reweight_demo.py
```

覆盖：

- `DoReMi`
- `ODM`
- `Loss Reweighting`

这个 demo 展示了：

- `DoReMi` 的 `Step 1 -> Step 2 -> Step 3` 风格流程
- `ODM` 的在线 domain weight 更新
- `Loss Reweighting` 的 trainer-aware sample weight 输出

```bash
PYTHONPATH=. python3 examples/policy_llamafactory_export_demo.py
```

这个 demo 会把策略筛过的数据导出成一个最小的 LLaMA-Factory bundle：

- `selected_messages.jsonl`
- `train.jsonl`
- `dataset_info.json`
- `llamafactory_sft.yaml`

```bash
PYTHONPATH=.:LLaMA-Factory/src python3 examples/policy_llamafactory_runtime_demo.py
```

这个 demo 会真实调用 vendored `LLaMA-Factory`，在训练里挂上 DataOS policy runtime，演示：

- `Loss Selector`
- `Loss Reweighting`
- `DoReMi`

并把运行产物写到：

- `examples/outputs/policy_llamafactory_runtime_demo/`

## 旧示例

下面这些脚本保留为原有 evaluator / mapper / selector 组合示例：

- `merge_to_alpaca_demo.py`
- `perplexity_topk_to_alpaca_demo.py`
- `ifd_topk_to_alpaca_demo.py`
- `deita_topk_to_alpaca_demo.py`
- `instag_coverage_to_alpaca_demo.py`
- `embedding_dedup_to_alpaca_demo.py`
- `cluster_balance_to_alpaca_demo.py`
- `llm_analysis_score_to_alpaca_demo.py`
- `llm_task_relevance_topk_to_alpaca_demo.py`
- `domain2vec_to_alpaca_demo.py`
