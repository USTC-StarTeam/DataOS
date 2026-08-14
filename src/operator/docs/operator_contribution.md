# 算子贡献指南

本文档说明新增一个算子时必须遵守的最小规则。

## 1. 选择父类

| 类型 | 父类 | 实现方法 |
| --- | --- | --- |
| Evaluator | `EvaluatorOperator` | `evaluate_record()` |
| Mapper | `MapperOperator` | `map_record()` |
| Refiner | `RefinerOperator` | `refine_record()` |
| Reweighter | `ReweighterOperator` | `reweight_record()` |
| Selector | `SelectorOperator` | `select()` |
| Generator | `GeneratorOperator` | `generate()` |
| Clusterer | `ClustererOperator` | `cluster()` |
| Mixer | `MixerOperator` | `mix()` |

record-first 算子可以处理 record，也可以处理 dataset。需要全局信息时，可以覆盖对应的 dataset 方法，例如 `evaluate_dataset()`。
`SelectorOperator`、`GeneratorOperator`、`ClustererOperator` 和 `MixerOperator` 都属于 dataset 级语义算子；调用方可以传单个 dataset 或多个 dataset，框架会在 hook 层统一成 `Sequence[Dataset]`。

## 2. 文件和布局

文件位置：

```text
dataos_operator/operators/<kind>/<operator_name>.py
```

布局规则：

- 文件名 stem 必须是 snake_case。
- 算子 kind 必须等于目录名。
- 一个模块只定义一个具体算子类。
- 算子类必须有 class docstring。

## 3. 最小示例

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator


class LengthFilter(SelectorOperator):
    """保留文本长度位于指定区间内的 records。"""

    def __init__(
        self,
        min_length: int,
        max_length: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(min_length=min_length, max_length=max_length, **kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                text = str(self.require_field(record, self.text_key))
                if self.min_length <= len(text) <= self.max_length:
                    selected.append(record)
        return selected
```

使用示例：

```python
operator = LengthFilter(text_key="text", min_length=3, max_length=8)
result = operator.run([{"text": "hello"}, {"text": "welcome"}])
```

注意：

- 不要重写 `run()`。
- 不要在具体方法里构造新的返回协议对象，只返回 record、records 或 `Dataset`。
- 自定义 `__init__()` 时，用 `**kwargs` 接收通用配置，并把本算子的配置一起传给 `super().__init__(...)`。
- 外部资源通过配置传入，不放进 `run()` 参数。
- dataset 级语义算子的 hook 要按“多个 dataset 的统一输入”来实现，即使调用方实际只传了一个 dataset。

## 4. 测试

新增算子必须补测试，建议放在：

```text
tests/operators/<kind>/test_<operator_name>.py
```

例如 `length_filter` 的测试放在：

```text
tests/operators/selector/test_length_filter.py
```

最小示例：

```python
from dataos_operator.operators.selector.length_filter import LengthFilter


def test_length_filter_selects_records_within_range() -> None:
    operator = LengthFilter(
        text_key="payload.text",
        min_length=3,
        max_length=5,
    )

    result = operator.run(
        [
            {"payload": {"text": "hi"}},
            {"payload": {"text": "hello"}},
        ]
    )

    assert result.to_records() == [{"payload": {"text": "hello"}}]
```

至少覆盖：

- 算子类能直接实例化。
- 配置参数能生效。
- 输出数据形态正确。
- dataset 级算子测试单个 dataset 和多个 dataset 两种输入。
- 有明确错误行为时测试异常。

## 5. 提交前检查

```bash
make check
```

CI 会执行 lint、format check、mypy 和 pytest。
