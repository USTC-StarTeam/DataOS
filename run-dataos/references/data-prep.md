# 数据阶段 —— load + 算子

判断全在这一阶段:用哪些源、什么格式、留多少。DataOS 只给你一个函数 `dataos.load`;之后全是
外部 `dataos_operator` 库,由你直接驱动(`Op(**params).run(ds) -> ds`)。**故意没有 DataOS
处理 API** —— 塑形数据的是 Claude,一个固定的 "process()" 只会碍事。

## 加载

`dataos.load(source, *, split="train", field_map=None, limit=None)` -> 一个
`dataos_operator.Dataset`,按 `source` 路由:

- `dataos.load("hf://openai/gsm8k", split="train", limit=100)` —— Hugging Face hub。
- `dataos.load("data/train.jsonl")` —— 本地 `.jsonl` / `.json` / `.csv`。
- `dataos.load("datasets/gsm8k", split="train", limit=8)` —— `save_to_disk` 目录
  (HF `DatasetDict`);自动选 split。⚠️ 自带的 `datasets/gsm8k` 带 `question` / `answer` /
  **`conversations`** 三列,见下面第 4 步**必须**的投影。
- `dataos_operator.Dataset.from_records([{...}])` —— 从内存行构建;冒烟测试最快的路。

## Dataset 接口(`dataos_operator.Dataset`,不是 HF Dataset)

`load` 返回的是 `dataos_operator.Dataset`,**不是** HF dataset。常用接口:

- `len(ds)` —— 行数(**不是** `ds.count`,那是 `Sequence.count(value)`,当行数用会报错)。
- `ds.to_records()` —— 取出 `list[dict]`。
- `ds.metadata` —— 元信息 dict。
- `ds.hf_dataset` —— 底层 HF `Dataset`,**列名 / 列数在这上面**(`ds.hf_dataset.column_names`;
  `Dataset` 本身没有 `.column_names`)。
- `ds.save_to_disk("<dir>")` —— 落盘(跨进程 / 并行用)。
- `Dataset.from_records(...)` / `Dataset.from_hf(...)` / `Dataset.from_local(...)` —— 构造。

## 一套可靠的配方

1. **要用时再发现算子。** `dataos.catalog()` -> 每个算子的 `name`、`kind`、`description`、
   `operator_class`、`params`(构造参数,含 `query_key` / `answer_key` 这类字段键)。命令行
   等价物是 `dataos catalog`。
2. **先归一到一种形状,再混。** 用 `OpenAIMessageMapper(source_format=...)`(认 `qa` /
   `alpaca` / `sharegpt` / `messages`)把每个源先映射成 OpenAI messages,再用
   `ConcatMixer().run([ds_a, ds_b, ...])` 合并。在同一套 schema 上混,避免字段错位的意外。
3. **按需过滤 / 采样。** 选择器在 `dataos_operator.operators.selector` 下 —— 如
   `LengthFilter(...)`、`RandomSampleSelector(sample_size=N, strict=False)`。
4. **映射到最终训练 schema。**
   - Alpaca:`AlpacaMapper(source_format="messages")` -> `instruction` / `input` / `output`。
   - ShareGPT:`ShareGPTMapper(source_format="messages")` -> `conversations`。
   训练阶段自动判别:**有 `conversations` 列 ⇒ ShareGPT,否则 Alpaca**。所以映射成 Alpaca 后,只要源里
   还残留 `conversations` 列(两个 mapper 都会原样透传它),就**必须**投影掉,否则会被误判成 ShareGPT:
   `FieldProjectMapper(keep_fields=("instruction","input","output"), strict=False)`。
   **自带的 `datasets/gsm8k` 就带 `conversations`——对它这步投影是必须的、不是可选的。**

## 跨实验复用

处理好的 `Dataset` 是个普通值 —— 多个实验共享数据时,**构建一次**复用即可。跨进程 / 并行时,
`ds.save_to_disk("<dir>")`,再把目录传给 `scripts/run_one.py` 的 `--data <dir>`(见
`references/execution.md`)。

## 已知训练形状(否则训练阶段会报错)

- Alpaca:`instruction` / `output`(可选 `input` / `system` / `history`)。
- ShareGPT:`conversations`(可选 `system` / `tools`)。

必填字段里出现非字符串的意外值会让训练阶段报错 —— 最终列保持这些形状。
