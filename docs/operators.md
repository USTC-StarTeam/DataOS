# 算子说明

本文档汇总仓库中当前已经实现的具体算子，按 `Evaluator`、`Mapper`、`Mixer` 和 `Selector` 分类说明。说明风格尽量贴近 API 文档，重点回答三个问题：算子做什么、`run(data)` 期望什么输入、会返回什么结果。

图片多模态算子的统一 schema、完整清单与推荐处理顺序见
[图片多模态基础算子](image_operators.md)。

所有算子的 `run(data)` 都接收 `Record` 或 `Dataset`（record 为普通 Python dict，`Dataset` 内部以 list-of-dicts 无损存储；Hugging Face/Arrow 仅作为导出形态），并返回新的 `Dataset`。

## 算子总览（按依赖类型）

按运行依赖可把算子分为三类：

- **纯规则（pure-rule）**：无外部依赖，纯 Python/numpy。
- **本地模型（local-model）**：按需加载 Hugging Face 模型（`transformers`/`torch`）。
- **在线（online）**：默认调用外部 LLM API（OpenAI 兼容，需 `api_key`/`base_url`），也可切 `backend="hf"` 本地后端。

| 算子 | kind | 依赖 |
| --- | --- | --- |
| `LengthFilter` | selector | 纯规则 |
| `ScoreSelector` | selector | 纯规则 |
| `RankingSelector` | selector | 纯规则 |
| `RandomSampleSelector` | selector | 纯规则 |
| `DeduplicateSelector` | selector | 纯规则 |
| `SetCoverageSelector` | selector | 纯规则 |
| `ClusterBalanceSelector` | selector | 纯规则（需前置聚类产出簇号） |
| `EmbeddingDeduplicateSelector` | selector | 纯规则（需前置 embedding 产出向量） |
| `AlpacaMapper` | mapper | 纯规则 |
| `ShareGPTMapper` | mapper | 纯规则 |
| `OpenAIMessageMapper` | mapper | 纯规则 |
| `FieldProjectMapper` | mapper | 纯规则 |
| `ConcatMixer` | mixer | 纯规则 |
| `Domain2VecMixer` | mixer | 本地模型（HF encoder + numpy） |
| `EmbeddingEvaluator` | evaluator | 本地模型（HF encoder） |
| `PerplexityEvaluator` | evaluator | 本地模型（HF 因果 LM） |
| `IFDEvaluator` | evaluator | 本地模型（HF 因果 LM） |
| `DeitaEvaluator` | evaluator | 本地模型（两个 HF 打分模型） |
| `InsTagEvaluator` | evaluator | 本地模型（HF 生成模型） |
| `EmbeddingClusterEvaluator` | evaluator | 本地模型（HF encoder + numpy） |
| `LLMAnalysisEvaluator` | evaluator | 在线 LLM（OpenAI 兼容，可切 hf） |
| `LLMTaskRelevanceEvaluator` | evaluator | 在线 LLM（OpenAI 兼容，可切 hf） |

## Evaluator

### `PerplexityEvaluator.run(data)`

为输入样本计算基于 response 的 perplexity 分数。输入为 `Record` 或 `Dataset`；输出为写入 perplexity 分数后的 `Dataset`。算子会优先读取样本中的 `messages` 字段构造上下文；若不存在 `messages`，则根据字段或模板构造 query 和 response，并只对 response 部分计算条件 perplexity，默认将结果写入 `metadata.perplexity`。

参数：
- `data`：待评分的数据输入。可以是单条 record，也可以是 dataset。算子会逐条处理样本；当 `score_key` 已存在且 `overwrite=False` 时直接复用已有分数；否则使用指定 Hugging Face 因果语言模型重新计算 perplexity 并写回样本。

返回：
- `Dataset`：带 perplexity 分数的样本集合。算子不会删除样本，只会在 record 中新增或更新分数字段；若样本既无法提供 `messages`，也无法构造 query/response，或 response 为空、消息结构非法，则抛出异常。

### `IFDEvaluator.run(data)`

为输入样本计算 instruction following difficulty, IFD 分数。输入为 `Record` 或 `Dataset`；输出为写入 IFD 分数后的 `Dataset`。算子会优先读取样本中的 `messages` 字段构造上下文；若不存在 `messages`，则根据字段或模板构造 query 和 response，并分别计算“带 query 的 response loss”和“不带 query 的 response loss”，再将二者比值作为 IFD 分数写回样本，默认写入 `metadata.ifd`。

参数：
- `data`：待评分的数据输入。可以是单条 record，也可以是 dataset。算子会逐条处理样本；当 `score_key` 已存在且 `overwrite=False` 时直接复用已有分数；否则使用指定 Hugging Face 因果语言模型重新计算 IFD 分数，并同时写回 `loss_with_query` 与 `loss_without_query` 供调试或分析。

返回：
- `Dataset`：带 IFD 分数的样本集合。算子不会删除样本，只会在 record 中新增或更新分数字段；若样本既无法提供 `messages`，也无法构造 query/response，或 response 为空、消息结构非法、`loss_without_query` 为 0，则抛出异常。

### `DeitaEvaluator.run(data)`

为输入样本计算 DEITA 风格的 Evol Complexity、Evol Quality 和 Evol Score。输入为 `Record` 或 `Dataset`；输出为写入 DEITA 分数后的 `Dataset`。算子会优先读取样本中的 `messages` 字段构造 instruction 与 response；若不存在 `messages`，则根据字段或模板构造 query 和 response，并分别计算复杂度分数、质量分数，再将二者乘积作为总分写入 `metadata.deita`。

参数：
- `data`：待评分的数据输入。可以是单条 record，也可以是 dataset。算子会逐条处理样本；当 `metadata.deita.score` 已存在且 `overwrite=False` 时直接复用已有结果；否则使用指定的 `complexity_model` 和 `quality_model` 重新计算，并写回 `metadata.deita.complexity`、`metadata.deita.quality` 和 `metadata.deita.score`。

返回：
- `Dataset`：带 DEITA 分数的样本集合。算子不会删除样本，只会在 record 中新增或更新 `metadata.deita`；若样本既无法提供 `messages`，也无法构造 query/response，或 instruction、response 为空，则抛出异常。

### `EmbeddingEvaluator.run(data)`

为输入样本计算文本 embedding，并将结果写入样本的 `metadata.embedding`。输入为 `Record` 或 `Dataset`；输出为写入 embedding 后的 `Dataset`。算子会优先读取样本中的 `messages` 字段构造文本；若不存在 `messages`，则优先使用 `text_template`，否则按 `query_key` 和 `answer_key` 组装文本。默认场景下，`query/answer` 会按 LlamaFactory 的 Alpaca 风格模板拼接后再计算 embedding。

参数：
- `data`：待编码的数据输入。可以是单条 record，也可以是 dataset。算子会逐条处理样本；当 `metadata.embedding.vector` 已存在且 `overwrite=False` 时直接复用已有 embedding；否则使用指定本地模型重新计算，并写回 `metadata.embedding.model`、`metadata.embedding.text`、`metadata.embedding.vector` 和 `metadata.embedding.dim`。

返回：
- `Dataset`：带 embedding 的样本集合。算子不会删除样本，只会在 record 中新增或更新 `metadata.embedding`；若样本既无法提供 `messages`，也无法通过模板或 `query/answer` 构造文本，或生成文本为空，则抛出异常。

### `InsTagEvaluator.run(data)`

为输入样本生成 instruction tags，并据此计算标签数量与复杂度分数。输入为 `Record` 或 `Dataset`；输出为写入 InsTag 结果后的 `Dataset`。算子会优先读取样本中的 `messages` 字段提取 instruction；若不存在 `messages`，则根据 `query_template` 或 `query_key` 构造 instruction，并使用指定模型生成标签结果，再将标准化后的标签、标签数、复杂度和原始输出写入 `metadata.instag`。

参数：
- `data`：待打标的数据输入。可以是单条 record，也可以是 dataset。算子会逐条处理样本；当 `metadata.instag.tags` 已存在且 `overwrite=False` 时直接复用已有结果；否则使用指定模型重新生成标签，并写回 `metadata.instag.tags`、`metadata.instag.tag_count`、`metadata.instag.complexity` 和 `metadata.instag.raw_output`。

返回：
- `Dataset`：带 instruction tags 的样本集合。算子不会删除样本，只会在 record 中新增或更新 `metadata.instag`；若样本既无法提供 `messages`，也无法通过模板或 `query_key` 构造 instruction，或生成 instruction 为空，则抛出异常。


### `EmbeddingClusterEvaluator.run(data)`

为输入样本生成聚类标签和轻量级簇信息。输入为 `Record` 或 `Dataset`；输出为写入 `metadata.cluster` 的 `Dataset`。算子会优先读取样本中的 `messages` 字段构造聚类文本；若不存在 `messages`，则优先使用 `text_template`，否则按 `query/answer` 或 `text` 字段构造文本。当前实现使用本地 embedding 模型生成向量，可选做 L2 归一化和 PCA 降维，再运行 `kmeans` 或 `minibatch_kmeans` 聚类；最终只把 `cluster.id`、`cluster.size`、`cluster.distance` 和 `cluster.rank` 写回 record，不会把完整向量写入 metadata。

参数：
- `data`：待聚类的数据输入。可以是单条 record，也可以是 dataset。算子会整体处理 dataset；当 `metadata.cluster.id` 已存在且 `overwrite=False` 时直接复用已有结果；否则重新计算 embedding、降维和聚类，并写回轻量级簇信息。

返回：
- `Dataset`：带聚类结果的样本集合。算子不会删除样本，只会在 record 中新增或更新 `metadata.cluster`；若样本既无法提供 `messages`，也无法通过模板或字段构造聚类文本，或 embedding 为空、维度非法，则抛出异常。

## Mapper

### `AlpacaMapper.run(data)`

将输入数据集中的样本映射为 Alpaca 训练格式。输入为 `Record` 或 `Dataset`；输出为映射后的 `Dataset`。算子会按 `source_format` 识别字段型样本或消息型样本，并将原始源字段转换为 `instruction`、`input`、`output`、`system`、`history` 等目标字段。

参数：
- `data`：待映射的数据输入。算子会逐条处理样本；当样本中存在被配置的源字段时，将其映射到 Alpaca 目标字段，并删除已消费的原始源键；当源字段不存在或值为 `None` 时，不创建对应目标键；若为消息型映射且消息结构不合法，则在 `strict=True` 时抛出异常。

返回：
- `Dataset`：映射后的样本集合。每条样本会保留未参与映射的原始字段，并新增解析得到的 Alpaca 字段；缺失字段不会补空字符串或空列表，只会跳过；被映射消费的原始字段会从结果样本中移除。

### `ShareGPTMapper.run(data)`

将输入数据集中的样本映射为 ShareGPT 对话格式。输入为 `Record` 或 `Dataset`；输出为映射后的 `Dataset`。算子会按 `source_format` 识别字段型样本或消息型样本，并将原始内容转换为 `conversations`、`system`、`tools` 等 ShareGPT 目标字段。

参数：
- `data`：待映射的数据输入。算子会逐条处理样本；当样本中存在被配置的源字段时，将其映射到 ShareGPT 目标字段，并删除已消费的原始源键；当源字段不存在或值为 `None` 时，不创建对应目标键；若为消息型映射且消息角色顺序或结构不合法，则在 `strict=True` 时抛出异常。

返回：
- `Dataset`：映射后的样本集合。每条样本会保留未参与映射的原始字段，并新增解析得到的 ShareGPT 字段；缺失字段不会补空值，只会跳过；被映射消费的原始字段会从结果样本中移除。

### `OpenAIMessageMapper.run(data)`

将常见监督微调样本映射为 OpenAI `messages` 格式。输入为 `Record` 或 `Dataset`；输出为映射后的 `Dataset`。算子支持字段型样本、问答型样本、已是 `messages` 的样本以及 ShareGPT `conversations` 样本，并统一转换为标准的 `messages` 列表。当前实现采用纯净转换：对于字段型样本，user message 直接使用原始 query 或 instruction；若 Alpaca 样本同时存在 `input`，则把 `instruction` 和 `input` 用换行拼接后写入同一个 user message，不再额外套 prompt 模板。

参数：
- `data`：待映射的数据输入。算子会逐条处理样本；当 `source_format="auto"` 时自动识别输入结构，优先判断 `messages`、其次 `conversations`、再判断 `query/answer`，否则按 Alpaca 字段处理；映射过程中会严格校验输入结构，并删除被消费的源字段，只保留新的 `messages` 以及未参与映射的其他字段。

返回：
- `Dataset`：映射后的样本集合。每条样本会生成标准的 `messages` 字段，元素采用 `{"role": ..., "content": ...}` 结构；若存在 `system` 或 `history`，会一并展开进消息列表；若输入结构非法、缺少所需字段、角色顺序不合法，或消息内容为空，则抛出异常。

### `FieldProjectMapper.run(data)`

按指定字段列表裁剪样本，只保留目标字段。输入为 `Record` 或 `Dataset`；输出为裁剪后的 `Dataset`。算子会按 `keep_fields` 读取 record 中的字段，并构造一个只包含这些字段的新 record，常用于在最终导出前去掉 `text`、`messages` 等中间字段，同时保留 `metadata`。

参数：
- `data`：待裁剪的数据输入。算子会逐条处理样本；当 `strict=True` 时，`keep_fields` 中任一字段缺失都会抛出异常；当 `strict=False` 时，缺失字段会被跳过。

返回：
- `Dataset`：仅保留目标字段后的样本集合。算子不会修改字段值，只改变 record 中保留哪些键；适合接在 `AlpacaMapper` 等映射算子之后，用于生成干净的导出 schema。


### `LLMAnalysisEvaluator.run(data)`

使用大模型从五个维度分析样本，并写入归一化总分。输入为 `Record` 或 `Dataset`；输出为写入 `metadata.llm_analysis` 的 `Dataset`。算子会优先读取 `messages`，否则按 `input_template`、`query/answer`、`text` 的顺序构造待分析内容，并要求模型返回包含五个维度分数的 JSON。

参数：
- `data`：待分析的数据输入。算子会逐条处理样本；当 `metadata.llm_analysis.score` 已存在且 `overwrite=False` 时直接复用已有结果；否则调用指定后端的大模型重新分析，并写回 `dimension_scores`、归一化 `score`、`label`、`reason` 和 `raw_output`。对于 API 后端，dataset 级运行默认使用 `concurrency=8` 并行请求。

返回：
- `Dataset`：带分析分数的样本集合。算子不会删除样本，只会在 record 中新增或更新 `metadata.llm_analysis`；若样本既无法提供 `messages`，也无法通过模板或字段构造内容，或模型输出不是预期 JSON，则抛出异常。

### `LLMTaskRelevanceEvaluator.run(data)`

使用大模型评估样本与目标任务描述的相关性，并写入归一化相关性分数。输入为 `Record` 或 `Dataset`；输出为写入 `metadata.llm_task_relevance` 的 `Dataset`。算子会优先读取 `messages`，否则按 `input_template`、`query/answer`、`text` 的顺序构造候选样本内容，并结合 `task_desc` 让模型输出 1 到 5 的相关性 JSON 结果。

参数：
- `data`：待评估的数据输入。算子会逐条处理样本；当 `metadata.llm_task_relevance.score` 已存在且 `overwrite=False` 时直接复用已有结果；否则调用指定后端的大模型重新分析，并写回归一化 `score`、`label`、`reason`、`task_desc` 和 `raw_output`。对于 API 后端，dataset 级运行默认使用 `concurrency=8` 并行请求。

返回：
- `Dataset`：带任务相关性分数的样本集合。算子不会删除样本，只会在 record 中新增或更新 `metadata.llm_task_relevance`；若样本无法构造输入内容、`task_desc` 为空或模型输出不是预期 JSON，则抛出异常。

## Mixer


### `Domain2VecMixer.run(data)`

根据目标数据集的 domain 分布估计 source datasets 的混合权重，并生成新的混合数据集。输入为一个或多个 source `Dataset`，外加单个或多个 `target_datasets`；输出为混合后的 `Dataset`。算子不会自己构造 domain，而是直接使用 `domain_key` 指向的已有标签，例如 `metadata.source` 或前置 `EmbeddingClusterEvaluator` 生成的 `metadata.cluster.id`。当前实现会先为每个 domain 估计 prototype embedding，再用 target_dataset 样本对各 domain 的相似度平均得到目标分布，并按该分布重采样 source 数据。

参数：
- `data`：待混合的 source 数据集输入。算子会统一遍历输入中的一个或多个 dataset；要求每条 source sample 都存在 `domain_key`；会根据 `messages`、`query/answer` 或 `text` 构造文本，使用本地 embedding 模型得到 domain prototype，再依据一个或多个 `target_datasets` 的平均 domain 概率估计 mixture weights。source 样本与 target 样本的分布估计采样规模可分别由 `source_sample_size` 与 `target_sample_size` 控制。

返回：
- `Dataset`：按目标 domain 分布重新混合后的样本集合。算子不会修改 record 内容，只改变最终保留的样本来源比例；dataset 级 metadata 中会附带 `target_distribution`、`mixture_weights` 等信息。若 source sample 缺失 `domain_key`，或 target/source 无法构造文本，或 embedding 模型不可用，则抛出异常。


### `ConcatMixer.run(data)`

按输入顺序直接拼接一个或多个数据集。输入为单个 `Dataset` 或由多个 `Dataset` 组成的序列；输出为拼接后的 `Dataset`。算子不会修改 record 内容，也不做字段对齐，只负责按顺序把样本合并在一起。

参数：
- `data`：待拼接的数据集输入。算子会按输入顺序依次遍历每个 dataset，并把所有样本合并成一个新的 dataset。

返回：
- `Dataset`：拼接后的样本集合。输出样本顺序与输入顺序一致；若要保证下游 evaluator 或 selector 的字段一致性，应先通过 mapper 把不同来源样本转换到统一 schema。

## Selector

### `LengthFilter.run(data)`

对输入数据集按文本长度范围进行筛选。输入为 `Dataset`，或由多个 `Dataset` 组成的序列；输出为筛选后的 `Dataset`。算子会读取每条样本中配置的文本字段，并根据设定的最小/最大长度判断是否保留。

参数：
- `data`：待筛选的数据集输入。算子会读取每条样本中配置的文本字段，并根据设定的最小/最大长度判断是否保留。若字段缺失且 `keep_missing=False` 则丢弃；若 `keep_missing=True` 则保留。当前 `length_unit` 只支持字符数。

返回：
- `Dataset`：仅包含满足长度条件样本的数据集；长度位于设定范围内的样本会被保留，不符合条件的样本会被过滤掉。

### `RandomSampleSelector.run(data)`

从输入数据集中随机选择固定数量的样本。输入为 `Dataset`，或由多个 `Dataset` 组成的序列；输出为随机采样后的 `Dataset`。

参数：
- `data`：待采样的数据集输入。算子会将输入中的样本汇总后，随机选择 `sample_size` 条样本返回；当可用样本数不足且 `strict=False` 时返回全部样本，否则抛出异常。

返回：
- `Dataset`：随机采样后的样本集合。

### `DeduplicateSelector.run(data)`

按指定字段值对输入数据集进行去重。输入为 `Dataset`，或由多个 `Dataset` 组成的序列；输出为去重后的 `Dataset`。算子会读取每条样本中配置的字段值，并根据 `keep` 策略保留首个或最后一个重复样本。

参数：
- `data`：待去重的数据集输入。算子会读取每条样本在 `key` 指向位置上的字段值；默认不区分大小写；可通过 `keep="first"` 或 `keep="last"` 控制保留策略。

返回：
- `Dataset`：去重后的样本集合；字段值相同的重复样本只保留一条，默认不区分大小写。

### `ScoreSelector.run(data)`

按指定分数字段的数值范围筛选样本。输入为 `Dataset`，或由多个 `Dataset` 组成的序列；输出为筛选后的 `Dataset`。算子会读取每条样本在 `score_key` 指向位置上的分数，并根据 `min_score`、`max_score` 决定是否保留。

参数：
- `data`：待筛选的数据集输入。算子会将输入中的一个或多个 dataset 统一遍历；当样本分数满足区间条件时保留；当分数字段缺失且 `keep_missing=False` 时丢弃，若 `keep_missing=True` 则保留；字符串形式的数值会自动转换为浮点数参与比较。

返回：
- `Dataset`：按分数字段筛选后的样本集合。只有满足 `min_score <= score <= max_score` 条件的样本会被保留；若只设置单侧阈值，则按单侧条件筛选；若 `min_score` 和 `max_score` 同时未设置，或区间非法，则在初始化时抛出异常。

### `RankingSelector.run(data)`

按一个或多个排序键选择 top-k 或 bottom-k 样本。输入为 `Dataset`，或由多个 `Dataset` 组成的序列；输出为筛选后的 `Dataset`。算子会读取每条样本在 `sort_keys` 指向位置上的数值字段，完成排序后保留前 `top_k` 或后 `bottom_k` 个样本。

参数：
- `data`：待筛选的数据集输入。算子会将输入中的一个或多个 dataset 统一遍历；若设置 `top_k`，则返回排序后的前 k 条；若设置 `bottom_k`，则返回后 k 条；`reverse=True` 时按降序理解“top”，`reverse=False` 时按升序理解“top”；当排序字段缺失且 `keep_missing=True` 时，缺失样本可在数量不足时补入结果。

返回：
- `Dataset`：按排序键筛选后的样本集合。算子不会修改 record 内容，只改变最终保留的样本子集；若排序字段不是数值，或 `top_k` / `bottom_k` 配置非法，则抛出异常。

### `SetCoverageSelector.run(data)`

按集合覆盖率与优先级贪心选择样本。输入为 `Dataset`，或由多个 `Dataset` 组成的序列；输出为筛选后的 `Dataset`。算子会读取每条样本在 `set_key` 指向位置上的集合字段，例如标签集合，并可结合 `priority_key` 对样本先排序，再依次选择能够带来新集合元素覆盖的样本。

参数：
- `data`：待筛选的数据集输入。算子会将输入中的一个或多个 dataset 统一遍历；若设置了 `priority_key`，则先按该数值字段排序；当前样本只有在新增集合元素数量不少于 `min_new_items` 时才会优先保留；若达到 `target_size` 或 `max_coverage_ratio`，则停止主选择流程；当 `fill_remaining=True` 时，会继续从未选样本中补足数量。

返回：
- `Dataset`：基于集合覆盖率选择后的样本集合。算子不会修改 record 内容，只改变最终保留的样本子集；若集合字段缺失且 `keep_missing=False` 则跳过，若 `keep_missing=True` 则可在补足阶段保留；若集合字段为空、格式非法，或优先级字段不是数值，则抛出异常。


### `ClusterBalanceSelector.run(data)`

按聚类标签为样本分配配额，并从每个簇中平衡选择子集。输入为 `Dataset`，或由多个 `Dataset` 组成的序列；输出为平衡采样后的 `Dataset`。算子会读取每条样本在 `cluster_key` 指向位置上的簇编号，并根据 `quota_strategy` 为每个簇分配样本数；簇内再按 `top_score`、`centroid_first` 或 `random` 选择具体样本。

参数：
- `data`：待筛选的数据集输入。算子会将输入中的一个或多个 dataset 统一遍历；当前支持 `uniform`、`sqrt` 和 `proportional` 三种簇间配额策略；若设置 `score_key`，则可以优先保留簇内高分样本；若设置 `intra_cluster_strategy="centroid_first"`，则会优先保留距离簇中心更近的典型样本。

返回：
- `Dataset`：平衡采样后的样本集合。算子不会修改 record 内容，只改变最终保留的样本子集；若簇编号字段缺失且 `keep_missing=False` 则跳过，若 `keep_missing=True` 则在补足阶段可保留；若簇编号非法、分数字段不可解析，或 `target_size` / `quota_strategy` 配置不合法，则抛出异常。

### `EmbeddingDeduplicateSelector.run(data)`

按 embedding 相似度过滤冗余样本。输入为 `Dataset`，或由多个 `Dataset` 组成的序列；输出为去重后的 `Dataset`。算子会读取每条样本在 `embedding_key` 指向位置上的向量，并按顺序贪心遍历：若当前样本与任一已保留样本的相似度达到阈值，则视为重复并丢弃，否则保留。

参数：
- `data`：待去重的数据集输入。算子会将输入中的一个或多个 dataset 统一遍历；当前版本使用 cosine 相似度判断冗余，可选对 embedding 先做 L2 归一化；当样本缺失 embedding 且 `keep_missing=False` 时丢弃，若 `keep_missing=True` 则保留；若设置 `max_samples`，则达到上限后停止继续选择。

返回：
- `Dataset`：基于 embedding 相似度去重后的样本集合。算子不会修改 record 内容，只改变最终保留的样本子集；当两个样本的相似度大于等于 `similarity_threshold` 时，后出现的样本会被视为重复而移除；若 embedding 维度不一致、为空或包含非数值内容，则抛出异常。

## 预留目录

以下类别目前只保留目录和 README，还没有具体算子实现：

- `generator`
- `clusterer`
- `reweighter`
- `refiner`

后续如果这些目录中新增了具体算子，建议继续按本文档同样的结构补充说明。
