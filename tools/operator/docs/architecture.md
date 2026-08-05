# 架构设计

本文档描述 DataOS Operator 的初始仓库设计。当前以框架结构和算子边界为主，只保留少量示例算子。


### `core`

`core` 保存跨算子共享的抽象契约，例如：

- 算子类别。
- 数据记录和数据集的基础类型。
- 算子元信息。
- 算子运行约定。
- 抽象算子接口和执行形态。
- 算子布局校验。

这里可以增加稳定、通用、低耦合的能力，但不放具体业务算子。

### `dataset.py`

根入口 `Dataset` 是数据接入对象，用于把本地 JSON、JSONL、CSV 或 Hugging Face 数据集转换成现有算子可直接接收的数据集。

它实现了 `Sequence[Record]`，因此可以直接传给：

```python
operator.run(dataset)
```

它只负责加载和字段标准化，不负责执行算子，也不引入 pipeline。加载本地文件或 Hugging Face
数据集时，每条 record 会自动补 `metadata.source`；本地文件默认取文件名去掉扩展名，其他来源默认取路径最后一段，也可以由调用方显式指定。

存储模型：`Dataset` 内部以 list-of-dicts 为标准存储，record 是任意异构 dict，类型、缺失字段都按原样无损 round-trip。Hugging Face `datasets`（Arrow，列式）只作为导出/落盘形态——`save_to_disk()` / `from_disk()` / `push_to_hf()` / `hf_dataset` 时才按需转换，`datasets` 仅在这些场景下 import（因此非强制依赖、也便于测试 mock）。之所以不用 Arrow 直接做内存存储：Arrow 要求每列单一类型、每行都有该列，会把首条记录缺失的字段整列丢弃、并对异构类型（如同一字段既有 `"2.5"` 又有 `1`）报错，与"record 是任意 dict"的契约冲突。

### `operators`

`operators` 按职责拆分目录：

- `evaluator`
- `generator`
- `mapper`
- `mixer`
- `clusterer`
- `reweighter`
- `refiner`
- `selector`

每个目录先保留 README 和空模块。后续新增具体算子时，应在对应目录中增加实现、配置说明和测试。

## 算子布局约定

具体算子直接定义在对应目录下：

```python
class RandomSelector(SelectorOperator):
    """随机选择 records。"""
```

第一版只强制以下稳定关系：

- 文件名 stem 使用 snake_case，例如 `random_selector.py`。
- 算子 kind 等于所在目录名，例如 `operators/selector/random_selector.py` 的 kind 是 `selector`。
- 每个具体算子模块只定义一个具体算子类。
- 算子类继承 `DataOperator`，并提供 class docstring。

类名不作为硬性约束，方便保留 `LLM`、`QA` 等缩写。

## 三层抽象

算子使用对象式主接口，内部可以自由拆函数。抽象分三层：

```text
DataOperator
├── RecordOperator
└── DatasetOperator
```

第一层 `DataOperator` 负责统一的 `run()` 和 dataset metadata。

第二层是执行形态层，限制的是**最细粒度操作**，不是限制算子只能处理这个粒度：

- `RecordOperator`：最细粒度是单条样本，可以直接处理 record，也可以处理 dataset。
- `DatasetOperator`：最细粒度是 dataset 级整体输入，具体语义算子可以接收单个 dataset 或多个 dataset。

第三层是语义算子层：

```text
RecordOperator
├── EvaluatorOperator
├── MapperOperator
├── RefinerOperator
└── ReweighterOperator

DatasetOperator
├── SelectorOperator
├── GeneratorOperator
├── ClustererOperator
└── MixerOperator
```

这四类 dataset 级语义算子在实现 hook 时统一接收 `Sequence[Dataset]`；如果调用方只传一个 dataset，框架会自动包装成长度为 1 的序列。

其中 `EvaluatorOperator`、`MapperOperator`、`RefinerOperator`、`ReweighterOperator`
默认都是 record-first，但都提供 dataset-level hook。例如 `EvaluatorOperator`
默认用 `evaluate_record()` map 出 `evaluate_dataset()`，需要全局上下文的 evaluator
可以覆盖 `evaluate_dataset()`。

`RecordOperator.run(record)` 和 `RecordOperator.run(dataset)` 最终都会返回同一个 `Dataset`
类，方便多步算子直接串联。`DataOperator.run()` 会把当前算子的名称、类别和输入输出数量等信息
写回返回的 `Dataset.metadata`，并把每一步运行信息追加到 `history`。

## 辅助输入

参考 Data-Juicer 的设计，主数据流保持为 dataset；辅助输入不作为任意位置参数散落在 `run()` 中，而优先通过以下方式进入算子：

- 算子初始化参数：例如模型名、prompt、采样参数、阈值、替换规则、输出字段名。
- record 字段：例如 `text_key`、`query_key`、`answer_key` 指向样本中的不同字段。
- 嵌套字段访问：例如 `metadata.score`、`messages.answer` 这类字段路径。
- 外部资源路径：例如 reference 文件、模型路径、taxonomy 文件，通过配置传入，再由算子内部加载。

因此当前基础架构先保留 `data` 作为主输入。后续如果确实需要统一传递运行期上下文，可以再增加 `context` 通道，而不是改变每个算子的主输入签名。

## 配置和字段路径

`DataOperator` 会保存初始化时传入的完整配置：

- `operator_config`：算子的完整初始化配置。
- `resource_config`：从配置中识别出的外部资源配置，例如 `model`、`prompt_path`、`reference_path`、`taxonomy_path`，以及以 `_path`、`_uri`、`_url` 结尾的字段。

如果存在 `resource_config`，`run()` 会把它写入返回 `Dataset` 的 `metadata["resources"]`，
方便后续追踪这个结果依赖了哪些模型、prompt 或参考文件。

常见字段名通过配置进入对象，例如 `text_key`、`query_key`、`answer_key`、`metadata_key`、`score_key`、`weight_key`。算子实现中可以使用：

```python
text = self.require_field(record, self.text_key)
record = self.set_field(record, self.output_key, score)
```

这里的 `output_key="metadata.eval.score"` 表示把输出写到样本的嵌套字段：

```python
{
    "metadata": {
        "eval": {
            "score": score
        }
    }
}
```

输入 record 不强制要求存在 `metadata` 字段。写入 `metadata.eval.score` 时，如果 `metadata`
或中间字段不存在，字段路径工具会自动创建。字段路径工具也支持读取已有 list 下标，例如
`messages.0.content`。写入字段时会返回新的 record，不直接修改原始 record。

### `docs/research`

这里用于承接论文、方法调研和复现记录。建议每篇论文一份 Markdown，包含：

- 论文链接或引用。
- 要解决的数据问题。
- 方法假设。
- 可实现成哪些算子。
- 输入输出和复杂度。
- 复现计划。

## 数据流约定

后续算子建议统一成类似形态：

```text
主数据输入 + 初始化参数 -> Dataset
```

其中：

- `RecordOperator` 的主数据输入可以是 `Record` 或 `Dataset`。
- `Dataset` 是样本序列。
- 算子初始化参数是对象配置。
- 输出主数据仍然是 `Dataset`；运行元信息写在 `Dataset.metadata`。

Evaluator、Clusterer、Reweighter 这类算子通常会给样本增加字段或 metadata；Selector 会减少样本；Generator 可能从空输入、种子输入或已有样本扩展出新样本；Mixer 会合并多个来源。
