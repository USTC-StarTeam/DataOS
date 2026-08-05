# 示例

这里存放一组可运行的算子流示例。所有流程都从原始数据集出发，先通过 `OpenAIMessageMapper` 统一成 OpenAI `messages` 格式，再做混合、筛选或去重，最后通过 `AlpacaMapper` 生成 Alpaca 字段，再通过 `FieldProjectMapper` 只保留 LlamaFactory 直接可用的 `instruction`、`input`、`output`，同时保留 record 内的 `metadata`。

输入数据集：

- `dataset/gsm8k/gsm8k_20.jsonl`
- `dataset/alpaca/alpaca_20.jsonl`

> 这两个输入文件位于被 `.gitignore` 忽略的 `dataset/` 目录，不随仓库分发；运行前需自行放置，或把脚本里的输入路径改成仓库自带的小样例数据（如 `tests/fixtures/gsm8k_target_100.jsonl`）。

输出目录：

- `examples/outputs/`

所有 demo 都只在最后打印一次结果摘要，包含输出路径、样本数、metadata 和首条样本。

## 可运行示例

```bash
PYTHONPATH=. python3 examples/merge_to_alpaca_demo.py
```

流程：`OpenAIMessageMapper` -> `OpenAIMessageMapper` -> `ConcatMixer` -> `LengthFilter` -> `RandomSampleSelector` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`

```bash
PYTHONPATH=. python3 examples/perplexity_topk_to_alpaca_demo.py
```

流程：`OpenAIMessageMapper` -> `OpenAIMessageMapper` -> `ConcatMixer` -> `PerplexityEvaluator` -> `RankingSelector` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`

```bash
PYTHONPATH=. python3 examples/ifd_topk_to_alpaca_demo.py
```

流程：`OpenAIMessageMapper` -> `OpenAIMessageMapper` -> `ConcatMixer` -> `IFDEvaluator` -> `RankingSelector` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`

```bash
PYTHONPATH=. python3 examples/deita_topk_to_alpaca_demo.py
```

流程：`OpenAIMessageMapper` -> `OpenAIMessageMapper` -> `ConcatMixer` -> `DeitaEvaluator` -> `RankingSelector` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`

```bash
PYTHONPATH=. python3 examples/instag_coverage_to_alpaca_demo.py
```

流程：`OpenAIMessageMapper` -> `OpenAIMessageMapper` -> `ConcatMixer` -> `InsTagEvaluator` -> `SetCoverageSelector` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`

```bash
PYTHONPATH=. python3 examples/embedding_dedup_to_alpaca_demo.py
```

流程：`OpenAIMessageMapper` -> `OpenAIMessageMapper` -> `ConcatMixer` -> `EmbeddingEvaluator` -> `EmbeddingDeduplicateSelector` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`

```bash
PYTHONPATH=. python3 examples/cluster_balance_to_alpaca_demo.py
```

流程：`OpenAIMessageMapper` -> `OpenAIMessageMapper` -> `ConcatMixer` -> `EmbeddingClusterEvaluator` -> `ClusterBalanceSelector` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`

```bash
PYTHONPATH=. python3 examples/llm_analysis_score_to_alpaca_demo.py
```

流程：`OpenAIMessageMapper` -> `OpenAIMessageMapper` -> `ConcatMixer` -> `LLMAnalysisEvaluator` -> `ScoreSelector` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`

```bash
PYTHONPATH=. python3 examples/llm_task_relevance_topk_to_alpaca_demo.py
```

流程：`OpenAIMessageMapper` -> `OpenAIMessageMapper` -> `ConcatMixer` -> `LLMTaskRelevanceEvaluator` -> `RankingSelector` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`

```bash
PYTHONPATH=. python3 examples/domain2vec_to_alpaca_demo.py
```

流程：切分后的 `gsm8k/alpaca-eval` source subsets -> `Domain2VecMixer(target_datasets=[gsm8k_test, ArenaHard_test, HumanEval_test])` -> `AlpacaMapper(always_include_input=True)` -> `FieldProjectMapper`
