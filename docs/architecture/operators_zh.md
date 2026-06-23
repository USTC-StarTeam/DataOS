# DataOS 算子体系（中文）

本文档说明 DataOS 当前的算子组织方式、每类算子的定位，以及为什么 plan
agent 现在能够稳定地产生参数并调用算子。

## 目录结构

算子由外部 `dataos_operator` 包提供，DataOS 通过
`src/dataos/core/catalog.py` 暴露结构化 catalog。算子通常按类别组织为：

- `mapper/`
- `selector/`
- `filter/`
- `evaluator/`
- `generator/`

每个具体算子的实现位于 `dataos_operator` 包中；本仓库只消费 catalog 和
执行接口。

## 当前内置算子

### mapper

- `map_to_alpaca`
  - 作用：把数据统一映射成 `instruction / input / output`
  - 适用场景：训练前统一格式，或者把混合列名数据清洗成 Alpaca 风格
- `map_to_sharegpt`
  - 作用：把数据统一映射成 `conversations`
  - 适用场景：希望保留多轮对话式表示，或者为对话模型准备 ShareGPT 风格数据

### selector

- `select_random_subset`
  - 作用：按固定随机种子抽取子集
  - 典型参数：`max_rows`、`seed`
- `select_top_by_score`
  - 作用：按已有质量分数字段选 Top-K
  - 典型参数：`score_column`、`max_rows`

### filter

- `filter_exact_dedup`
  - 作用：按若干字段组合 key 做精确去重
  - 典型参数：`keys`
- `filter_by_text_length`
  - 作用：过滤过短或过长样本
  - 典型参数：`min_words`、`max_words`

### evaluator

- `evaluate_text_quality`
  - 作用：用启发式规则打质量分
  - 输出：默认写回 `quality_score`、`quality_reason`
- `evaluate_llm_pair`
  - 作用：调用 OpenAI-compatible 模型对问答对打分
  - 输出：默认写回 `llm_quality_score`、`llm_quality_reason`

### generator

- `generate_rewritten_query`
  - 作用：利用大模型改写 query / prompt，并把新样本追加回数据集
- `generate_rewritten_answer`
  - 作用：利用大模型增强或重写 answer，并把新样本追加回数据集

## 算子是如何被 plan agent 理解的

每个算子现在除了 `name` 和 `description`，还会暴露结构化元信息：

- `category`
- `description`
- `chinese_description`
- `parameters`
- `default_params`

其中 `parameters` 里明确包含：

- 参数名
- 类型
- 是否必填
- 默认值
- 可选值
- 参数说明

plan agent 拿到的是完整 schema，而不是只有名字的列表；因此它可以：

1. 知道算子属于哪一类。
2. 知道哪些参数必填。
3. 在任务没有明确约束时直接使用默认参数。
4. 在 plan 校验阶段把参数做一次标准化和类型转换。

## 为什么现在更容易跑通

这次改造重点不是“多写几个算子”，而是把链路打通：

1. registry 提供结构化算子描述。
2. prompt 把这些描述完整交给 LLM。
3. plan 校验按算子 schema 规范化参数。
4. operator 执行后稳定产出新的 `DatasetArtifact`。
5. artifact metadata 保留 `dataset_format`、父子 lineage 和算子执行信息。
6. 训练导出阶段可以识别 Alpaca 和 ShareGPT 两种数据表示。

## 数据格式约定

- DataOS 内部仍然只接受 Hugging Face `Dataset` / `DatasetDict` 的
  `save_to_disk` 目录作为 `DatasetArtifact`。
- `map_to_alpaca` 产出 `instruction / input / output`
- `map_to_sharegpt` 产出 `conversations`
- 训练阶段消费 `map_to_alpaca` 等数据算子已经产出的 Alpaca-like 数据。
