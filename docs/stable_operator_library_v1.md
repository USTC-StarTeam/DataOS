# 稳定算子库 V1

这份文档描述了 `dataos/operators/` 下第一批手写的稳定算子。
它是面向算子工程师的配套说明，可与 `docs/dataos_overview.md` 对照阅读。

## 设计立场

这些算子刻意**不是**对 DataJuicer 或 DataFlow 的直接照搬。
它们体现 DataOS 的原生取向，主要有三点：

1. 它们是 **artifact-first** 的：每个算子都消费 `DatasetArtifact`，并产出一个带 lineage 的全新 `DatasetArtifact`。
2. 它们是 **split-aware** 的：`DatasetDict` 是默认心智模型，而不是事后补上的兼容层。
3. 它们是 **research-loop-oriented** 的：元数据、日志和确定性行为，与具体变换本身同样重要。

我们的目标不是堆出一个最大的算子动物园，而是提供一套紧凑、可审计、可复现、能够稳定嵌入 DataOS 守护进程式研究闭环中的算子集合。

## 算子家族

### 1. 修复 / 治理

| 算子 | 作用 | DataOS 特性 |
|---|---|---|
| `schema_guard` | 把结构不统一的 SFT 样本规整为标准的 `instruction/input/response` 字段。 | 把 schema 修复视为稳定算子的一等能力，而不是隐藏在预处理胶水代码里的临时逻辑。 |
| `pii_redact` | 从文本列里脱敏明显的邮箱 / 电话 / URL。 | 保持数据集整体形状不变，并把脱敏计数写进 artifact 元数据，方便后续审计。 |

### 2. 低成本过滤

| 算子 | 作用 | DataOS 特性 |
|---|---|---|
| `length_filter` | 丢弃长度超出简单阈值范围的样本。 | 同时支持总长度阈值和按列阈值，并保持 Hugging Face dataset 原生工作方式。 |
| `lang_filter` | 只保留主导书写系统属于 `en`、`zh`、`mixed` 等允许标签的样本。 | 采用轻量、split-aware 的脚本启发式，而不是依赖外部语言服务。 |
| `quality_filter` | 通过启发式打分或可选的 LLM 评审，移除较弱的 instruction-response 对。 | 在 artifact 旁显式落盘 `quality_judgments.jsonl` 审计轨迹。 |

### 3. 标注 / 均衡 / 排序

| 算子 | 作用 | DataOS 特性 |
|---|---|---|
| `category_tag` | 打上粗粒度任务表面标签，例如 `translation`、`math`、`code`、`writing`。 | 设计目标是为后续均衡与分析算子提供可控信号。 |
| `category_balance` | 按类别对某个 split 做确定性的下采样 / 上采样平衡。 | 平衡完成后会重新交错类别，避免结果 split 呈现明显分块。 |
| `curriculum_sort` | 按简单表面复杂度代理分数，把样本从易到难或从难到易排序。 | 会显式写出 curriculum score 列，方便下游分析检查。 |

### 4. 数据构造 / 混合

| 算子 | 作用 | DataOS 特性 |
|---|---|---|
| `instruction_synthesis_template` | 在保留 gold response 的前提下，生成确定性的 instruction 改写版本。 | 追加 `dataos_synth_template`、`dataos_synth_source_index` 等 provenance 列。 |
| `mixture` | 从多个上游数据集 artifact 中构建训练混合数据。 | 直接基于 artifact id 工作，并把来源 provenance 写进每一条混合样本。 |

## 算子概要

### `schema_guard`

- 规范化常见别名，比如 `prompt -> instruction`、`output -> response`。
- 可选的 strict 模式可以在发现问题时快速失败，而不是静默修复。
- 最适合放在 pipeline 前段，用来处理异构数据源。

### `pii_redact`

- 支持邮箱、电话和 URL 脱敏。
- 既可以原位遮蔽 PII，也可以直接丢弃含 PII 的样本。
- 适合放在数据合成、导出或外部评审之前。

### `length_filter`

- 支持 `words` 或 `chars` 两种计量单位。
- 可以同时限制总长度和按列长度。
- 适合在训练或评审前裁掉异常长尾样本。

### `lang_filter`

- 支持 `allowed_languages=[...]`，例如 `['en']`、`['zh']`、`['mixed']`。
- 可选 annotation 模式可以把检测到的标签写回某一列。
- 它的定位是廉价路由过滤器，不是基准级语言识别器。

### `quality_filter`

- `mode='heuristic'`：完全本地、确定性的质量打分。
- `mode='llm_judge'`：调用 OpenAI-compatible `/chat/completions` 接口，并要求 JSON 输出。
- 在 LLM 模式下，使用标准参数，如 `base_url`、`api_key`、`model`。

### `category_tag`

- 采用简单的任务表面预设，而不是重量级语义分类器。
- 它提供的是低成本结构信号，供其他算子消费。
- 尤其适合放在 `category_balance` 之前。

### `category_balance`

- 支持 `downsample_to_min`、`cap` 和 `upsample_to_max`。
- 一次只平衡一个 split，通常是 `train`。
- 其他 split 保持不动。

### `curriculum_sort`

- 依据 prompt/response 长度、格式特征、代码标记等信息计算一个简单的表面复杂度代理分数。
- 它面向的是确定性 curriculum 实验，而不是追求完美难度估计。
- 结果分数会写入一个可配置列。

### `mixture`

- 基于 artifact id 混合多个父级 artifact。
- 每一条混合样本都会写入 `dataos_source_artifact` provenance。
- 它刻意保持小而确定，不把自己扩展成一个泛化 recipe 语言。

### `instruction_synthesis_template`

- 从种子训练 split 中生成小规模、确定性的 instruction 变体。
- 第一版刻意避免隐藏的 LLM 成本。
- 它适合作为廉价基线，与未来的 `self_instruct_lite` 或候选算子进行对照。

## 为什么它和 DataJuicer / DataFlow 不一样

### 相比 DataJuicer

- DataJuicer 更强调覆盖广泛数据工程场景的通用算子动物园。
- DataOS 的稳定算子更窄、更有观点：它们优先优化 SFT 研究闭环、artifact lineage，以及 split-aware 实验。
- DataOS 算子会显式记录紧凑但足够明确的运行元数据，因为下游消费者不只是变换后的数据本身，还包括 analyzer / critic / report 闭环。

### 相比 DataFlow

- DataFlow 风格的 recipe 往往把数据步骤视为更大 pipeline 系统里的工作流节点。
- DataOS 的稳定算子刻意做得更小粒度，并直接接入 `run_task()` 的算子规划阶段。
- 在 DataOS 里，关键抽象是 typed artifact chain，而不是通用 DAG DSL。

## 推荐的 pipeline 模式

### 最小清洗模式

1. `schema_guard`
2. `pii_redact`
3. `lang_filter`
4. `quality_filter`

### 类别感知模式

1. `schema_guard`
2. `category_tag`
3. `category_balance`
4. `curriculum_sort`

### 低成本增强模式

1. `schema_guard`
2. `quality_filter`
3. `instruction_synthesis_template`
4. `mixture`

## 后续工作说明

- 未来可以在同一套 artifact contract 之上增加 `self_instruct_lite`，但必须明确写清模型成本和 provenance。
- 未来 `quality_filter` 的 judge pool 可以在研究守护进程开始管理 token backpressure 之后，更积极地做 OpenAI-compatible 请求批处理。
- 候选算子在被验证并晋升之前，应继续与这套稳定算子库分开管理。
