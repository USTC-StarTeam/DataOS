# 数据阶段 —— DataStage(load + 算子 recipe)

判断全在这一阶段:用哪些源、什么格式、留多少。`exp.data` 是一个 **DataStage**:它 load 一个源、跑一串
**确定性算子 recipe**,产出一等的 `Data` artifact(`save_to_disk` 到 `dataset/`,`meta` 记 source / recipe /
rows / columns,`inputs` 记血缘)。算子来自外部 `dataos_operator` 库——**故意没有 DataOS 处理 API**,塑形
数据的逻辑由你定;一个固定的 "process()" 只会碍事。

## 两种用法

**1) 声明式 recipe(推荐)** —— 把数据计划写进 `spec["data"]`,再 `exp.data.run()`:

```python
spec["data"] = {
    "source": "datasets/gsm8k", "split": "train", "limit": 200,
    "field_map": None,                       # load 时可选的字段重命名
    "recipe": [                              # 线性、单数据集;每步 = OpClass(**params).run(ds) -> ds
        {"op": "openai_message_mapper", "params": {"source_format": "qa"}},
        {"op": "alpaca_mapper",         "params": {"source_format": "messages"}},
        {"op": "field_project_mapper",  "params": {"keep_fields": ["instruction", "input", "output"], "strict": False}},
    ],
}
exp.data.run()   # -> Data;算子按 op 名解析(snake spec 名或类名都认),未知名直接报错并列出已知名
```

**2) 逃生口 `set()`** —— 多源 mixing、或线性 recipe 表达不了的处理:自己用算子搭好 `Dataset` 再注入:

```python
ds = ConcatMixer().run([ds_a, ds_b])     # 多源混合不在线性 recipe 内 —— 走逃生口
exp.data.set(ds).run()                    # 注入现成 Dataset,跳过 recipe(meta.source="<injected>")
```

跑完用 `exp.data.artifact`(`Data`)/ `exp.data.summary()` 拿产物与状态;`Data.load()` 可重新读回 `Dataset`。

## 加载(`dataos.load`)

`dataos.load(source, *, split="train", field_map=None, limit=None)` -> 一个 `dataos_operator.Dataset`,
按 `source` 路由(recipe 模式下,DataStage 内部就是用它加载 `spec["data"]["source"]` 的):

- `dataos.load("hf://openai/gsm8k", split="train", limit=100)` —— Hugging Face hub。
- `dataos.load("data/train.jsonl")` —— 本地 `.jsonl` / `.json` / `.csv`。
- `dataos.load("datasets/gsm8k", split="train", limit=8)` —— `save_to_disk` 目录(HF `DatasetDict`);
  自动选 split。⚠️ 自带的 `datasets/gsm8k` 带 `question` / `answer` / **`conversations`** 三列,见下面
  **必须**的投影。
- `dataos_operator.Dataset.from_records([{...}])` —— 从内存行构建;冒烟测试最快的路(配 `set()` 逃生口)。

## Dataset 接口(`dataos_operator.Dataset`,不是 HF Dataset)

`load` 返回的是 `dataos_operator.Dataset`,**不是** HF dataset。常用接口:

- `len(ds)` —— 行数(**不是** `ds.count`,那是 `Sequence.count(value)`,当行数用会报错)。
- `ds.to_records()` —— 取出 `list[dict]`。
- `ds.metadata` —— 元信息 dict。
- `ds.hf_dataset` —— 底层 HF `Dataset`,**列名 / 列数在这上面**(`ds.hf_dataset.column_names`;
  `Dataset` 本身没有 `.column_names`)。
- `ds.save_to_disk("<dir>")` —— 落盘(DataStage 自动做;手动跨进程也可用)。
- `Dataset.from_records(...)` / `Dataset.from_hf(...)` / `Dataset.from_local(...)` —— 构造。

## recipe 里放什么(一套可靠的配方)

下面每一步都对应 recipe 里的一个 `{op, params}`(或逃生口里的一次 `Op(**params).run(ds)`):

1. **要用时再发现算子。** `dataos.catalog()` -> 每个算子的 `name`、`kind`、`description`、
   `operator_class`、`params`(构造参数,含 `query_key` / `answer_key` 这类字段键)。recipe 的 `op` 用
   `name`(snake)或类名都行;`params` 就是这里的构造参数。命令行等价物 `dataos catalog`。
2. **先归一到一种形状,再混。** `openai_message_mapper`(`source_format` 认 `qa` / `alpaca` /
   `sharegpt` / `messages`)把每个源先映射成 OpenAI messages,再 `ConcatMixer().run([ds_a, ds_b, ...])`
   合并。⚠️ **混多个源是多数据集操作,不能写进线性 recipe** —— 用 `set()` 逃生口手搭后注入。
3. **按需过滤 / 采样。** 选择器在 `dataos_operator.operators.selector` 下 —— 如
   `{"op": "length_filter", ...}`、`{"op": "random_sample_selector", "params": {"sample_size": N, "strict": False}}`。
4. **映射到最终训练 schema。**
   - Alpaca:`{"op": "alpaca_mapper", "params": {"source_format": "messages"}}` -> `instruction` / `input` / `output`。
   - ShareGPT:`{"op": "sharegpt_mapper", "params": {"source_format": "messages"}}` -> `conversations`。
   训练阶段自动判别:**有 `conversations` 列 ⇒ ShareGPT,否则 Alpaca**。所以映射成 Alpaca 后,只要源里还残留
   `conversations` 列(两个 mapper 都会原样透传它),就**必须**投影掉,否则会被误判成 ShareGPT:
   `{"op": "field_project_mapper", "params": {"keep_fields": ["instruction","input","output"], "strict": False}}`。
   **自带的 `datasets/gsm8k` 就带 `conversations`——对它这步投影是必须的、不是可选的。**

## 跨实验复用

处理好的 `Dataset` 是个普通值 —— 多个实验共享同一份数据时,**构建一次**复用即可:用 `set()` 逃生口注入,
跨进程 / 并行则 `ds.save_to_disk("<dir>")` 后把目录传给 `scripts/run_one.py` 的 `--data <dir>`(脚本内部用
`exp.data.set(...)` 注入,跳过按 spec 重跑 recipe)。见 `references/execution.md`。

## 已知训练形状(否则训练阶段会报错)

- Alpaca:`instruction` / `output`(可选 `input` / `system` / `history`)。
- ShareGPT:`conversations`(可选 `system` / `tools`)。

必填字段里出现非字符串的意外值会让训练阶段报错 —— 最终列保持这些形状。
