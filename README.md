# DataOS Operator

这是 [DataOS](https://github.com/USTC-StarTeam/DataOS) 的算子实现仓库。

## 仓库构成

```text
dataos_operator/
  dataset.py      # 本地路径 / Hugging Face 路径到标准 Dataset 的接入对象
  core/          # 算子抽象、字段工具、算子说明生成和贡献规范校验
  operators/     # 具体算子实现，按 evaluator / selector / mixer 等职责分类

examples/        # DataOS 可参考的 few-shot 使用示例
dataset/         # 仓库级原始或导出的数据文件，优先保存为 jsonl
tests/           # 单测和算子贡献规范检查
docs/            # 架构、贡献指南、TODO 和调研记录
configs/         # 后续放单算子配置示例
scripts/         # 后续放仓库辅助脚本
```

当前仓库只保留轻量框架和少量示例算子，不包含 pipeline 层。

## 安装与导入

其他项目可以直接从 GitHub 安装：

```bash
uv add "dataos-operator @ git+https://github.com/USTC-StarTeam/DataOS-Operator.git"
```

如果需要 Hugging Face 数据集接入能力，安装 `hf` 可选依赖：

```bash
uv add "dataos-operator[hf] @ git+https://github.com/USTC-StarTeam/DataOS-Operator.git"
```

如果需要运行基于 Hugging Face 模型的 evaluator，例如 `PerplexityEvaluator`，还需要安装 `llm` 可选依赖：

```bash
uv add "dataos-operator[llm] @ git+https://github.com/USTC-StarTeam/DataOS-Operator.git"
```

如果你是在仓库内本地开发，推荐按需要同步：

```bash
uv sync --extra dev --extra hf --extra llm
```

安装包名是 `dataos-operator`，Python 导入包名是 `dataos_operator`：

```python
from dataos_operator import Dataset
from dataos_operator.operators.selector.length_filter import LengthFilter

dataset = Dataset.from_records([{"text": "hi"}, {"text": "hello"}])
result = LengthFilter(text_key="text", min_length=3, max_length=5).run(dataset)

print(result[0])
print(dict(result.metadata))
```

## 算子说明

DataOS 可以通过算子说明了解当前仓库有哪些可用算子。算子说明由具体算子类自动生成，来源包括：

- `name`：算子配置名，默认等于文件名 stem，例如 `length_filter`。
- `kind`：算子类别，例如 `evaluator`。
- `description`：算子类 docstring，统一采用「一句话作用 + `Implementation:` + `Dependency:`」结构，其中 `Dependency:` 标明该算子是纯规则（pure-rule）、需本地模型（local-model），还是在线 LLM API（online）。
- `operator_class`：完整 Python 类路径。

通过 `list_operator_specs()` 获取（见下方代码），输出形态：

```json
[
  {
    "name": "length_filter",
    "kind": "selector",
    "description": "Filter samples by text length, keeping records whose length falls within the range.\n\nImplementation: ...\nDependency: pure-rule operator, no external dependencies.",
    "operator_class": "dataos_operator.operators.selector.length_filter.LengthFilter"
  }
]
```

代码中也可以直接获取：

```python
from dataclasses import asdict

from dataos_operator.core import list_operator_specs

operator_specs = [asdict(spec) for spec in list_operator_specs()]
```

## 数据接入

算子接收的主输入是 `Dataset`。仓库提供根入口对象 `Dataset`，用于把本地文件或 Hugging Face 数据集构造成算子可直接接收的 dataset 对象。
`Dataset.from_local()` 和 `Dataset.from_hf()` 会在每条 record 的 `metadata.source`
里自动补来源名；本地文件默认取文件名去掉扩展名，其他来源默认取路径最后一段，也可以通过 `source_name=` 显式指定。

`Dataset` 内部以 list-of-dicts 作为标准存储：每条 record 都是普通 Python dict，任意异构字段、缺失字段、混合类型都会被无损保留，算子和字段路径工具完全按 dict 处理。Hugging Face `datasets`（Arrow，列式）只作为**导出/落盘**形态——仅在 `save_to_disk()` / `from_disk()` / `push_to_hf()` / `hf_dataset` 时才按需转换，因此 `datasets` 也只在这些场景下才会被 import。

读取本地 JSONL：

```python
from dataos_operator import Dataset
from dataos_operator.operators.selector.length_filter import LengthFilter

dataset = Dataset.from_local(
    "tests/fixtures/gsm8k_target_100.jsonl",
    limit=100,
    source_name="gsm8k-demo",
)

result = LengthFilter(text_key="text", min_length=80, max_length=180).run(dataset)
```

读取 Hugging Face 数据集：

```python
from dataos_operator import Dataset

dataset = Dataset.from_source(
    "hf://openai/gsm8k",
    name="main",
    split="train[:100]",
    field_map={
        "text": "question",
        "answer": "answer",
    },
    target_regex=r"####\s*(.+)$",
)
```

使用 Hugging Face 接入时需要安装可选依赖：

```bash
uv sync --extra hf --extra dev
```

保存到本地：

```python
dataset.save_local("outputs/gsm8k_sample.jsonl")
dataset.save_local("outputs/gsm8k_sample.json")
dataset.save_local("outputs/gsm8k_sample.csv")
```

上传到 Hugging Face Hub：

```python
dataset.push_to_hf(
    "your-name/gsm8k-sample",
    split="train",
    config_name="default",
)
```

真实上传测试默认不会运行。需要先设置 `HF_TOKEN`，再显式开启：

```bash
RUN_HF_INTEGRATION=1 uv run --extra dev --extra hf pytest tests/integration/test_hf_dataset_upload.py
```

该测试会创建一个临时私有 dataset repo，上传完成后在 `finally` 中删除。

## 示例

`examples/` 下是一组完整的算子流示例（few-shot），可供 DataOS 学习算子的输入、实例化方式、输出和 metadata。每条流程都从原始数据集出发，先经 `OpenAIMessageMapper` 统一成 messages，再做混合 / 评分 / 筛选 / 去重，最后用 `AlpacaMapper` + `FieldProjectMapper` 导出 LlamaFactory 可用的 Alpaca JSONL。完整清单和每条流程见 [examples/README.md](examples/README.md)。

运行合并 + 随机采样导出示例（纯规则，无需模型）：

```bash
PYTHONPATH=. python3 examples/merge_to_alpaca_demo.py
```

运行基于本地小模型的 perplexity 打分 + top-k 示例（需 `--extra llm` 安装 torch/transformers）：

```bash
PYTHONPATH=. uv run --extra llm python examples/perplexity_topk_to_alpaca_demo.py
```

> 注意：这些示例默认读取 `dataset/gsm8k/gsm8k_20.jsonl`、`dataset/alpaca/alpaca_20.jsonl`。`dataset/` 目录被 `.gitignore` 忽略、不随仓库分发，运行前需自行放置对应输入文件，或把脚本里的输入路径改成仓库自带的小样例数据（如 [tests/fixtures/gsm8k_target_100.jsonl](tests/fixtures/gsm8k_target_100.jsonl)）。

## 仓库数据

根目录的 `dataset/` 约定用于存放仓库级数据文件（例如从 `openai/gsm8k`、`tatsu-lab/alpaca_eval` 导出的 `jsonl`），供示例和后续处理作为输入。该目录被 `.gitignore` 忽略、**不随仓库提交**，需要在本地自行放置；仓库内置的小样例数据见 [tests/fixtures/](tests/fixtures/)。

## 开发检查

```bash
uv sync --extra dev
make check
```

CI 会执行 ruff、format check、mypy 和 pytest，其中 pytest 会检查算子目录布局和贡献规范。

## 文档

- [架构设计](docs/architecture.md)
- [算子贡献指南](docs/operator_contribution.md)
- [TODO](docs/TODO.md)
- [调研记录](docs/research/README.md)
