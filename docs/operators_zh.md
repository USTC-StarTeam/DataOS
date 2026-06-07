# 算子(外部 dataos-operator)

DataOS 本身**不再内置算子**。所有 map / filter / sample / mix 等算子都来自外部包
[`dataos-operator`](https://github.com/USTC-StarTeam/DataOS-Operator),数据集类型也是它的
`dataos_operator.Dataset`(HF Arrow 背书)。

## 怎么发现和用

- **发现**:`dataos.catalog()` 返回每个算子的 `name` / `kind` / `description` / `operator_class` /
  `params`(构造参数,含 `query_key` / `answer_key` 等字段键)。命令行:`uv run dataos catalog`。
- **使用**:由 Claude 直接驱动 —— `Op(**params).run(ds) -> ds`。DataOS **没有**处理 API,这是有意的:
  塑形数据的判断属于 Claude(runtime / planner)。

## 一套常见配方

1. 用 `OpenAIMessageMapper(source_format=...)` 把各源归一成 OpenAI messages,再
   `ConcatMixer().run([ds_a, ds_b, ...])` 混。
2. 用 selector(`LengthFilter`、`RandomSampleSelector(sample_size=N)` 等)过滤 / 采样。
3. 用 `AlpacaMapper` / `ShareGPTMapper` 映射到最终训练 schema:
   - Alpaca = `instruction` / `input` / `output`
   - ShareGPT = `conversations`
   训练阶段自动判别:有 `conversations` 列 ⇒ ShareGPT,否则 Alpaca。

更细的数据阶段指引见 skill 的
[`run-dataos/references/data-prep.md`](../run-dataos/references/data-prep.md)。
