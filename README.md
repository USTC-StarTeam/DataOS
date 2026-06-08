# DataOS

DataOS 是一个本地优先的 SFT 数据流水线**确定性执行器库**——加载、处理、训练、评估、复盘都由本地
Python 完成,**不调用 LLM**。规划和判断交给 **Claude Code**(runtime / planner):它读 skill、写计划、
命令式驱动各阶段。DataOS 自己没有内嵌 agent、没有一把梭的 `run()`、也没有调度器。

- 架构与用法详解见仓库内的 skill:[`run-dataos/SKILL.md`](run-dataos/SKILL.md)。
- 算子来自外部库 [`dataos-operator`](https://github.com/USTC-StarTeam/DataOS-Operator)。

## 范围

- 两个容器:`Study`(多实验)→ `Experiment`(单次)。**`Study` 就是一个目录**,`compare()` 读盘聚合,
  所以并行与否由 runtime 决定,不归 DataOS。
- 数据集是外部 `dataos_operator.Dataset`(HF Arrow 背书),不是 DataOS 自己的类型。
- `train`(LLaMA-Factory)真透传任意键(无白名单);`eval`(lm-eval)用 `EvalConfig` 字段集
  (覆盖 lm-eval 常用键,未知键忽略 + 告警)。
- 一切落盘:`<workspace>/<study_id>/{plan.json, review.json, experiments/<name>/.../report.json}`。
- 没有研究层、守护进程、项目队列或持久 agent 控制面。

## 安装

```bash
uv sync                  # 核心(stub 全流程够用)
uv sync --extra train    # 真实训练(llamafactory + torch + GPU)
uv sync --extra eval     # 真实评测(lm-eval)
```

核心依赖:`datasets`、`pyarrow`、`dataos-operator`(git)。

## 怎么用

DataOS 是给 **Claude Code** 用的库,不是给人手写脚本的。把仓库的 skill 注册给 Claude Code
(`~/.claude/skills/` 或项目 `.claude/skills/`),然后用自然语言说目标,Claude 会照 skill 驱动它。
详见 [`run-dataos/SKILL.md`](run-dataos/SKILL.md)。

最小冒烟(stub,无 GPU / 无 LLM key):

```python
import dataos
from dataos_operator import Dataset

study = dataos.study(id="smoke")
study.set_plan({"goal": "冒烟", "experiments": [
    {"name": "a", "train": {"mode": "stub"},
     "eval": {"provider": "stub", "provider_config": {"score": 0.1}}},
]})
for spec in study.plan["experiments"]:
    exp = study.experiment(spec)
    ds = Dataset.from_records(
        [{"instruction": "q", "input": "", "output": "a"}],
        metadata={"dataset_format": "alpaca"})
    exp.train(ds); exp.eval(); exp.finalize()
print(study.compare())
study.finalize(conclusion="done")
```

环境自检:`python run-dataos/scripts/doctor.py`。

## CLI

只有一个发现用的子命令:

```bash
uv run dataos catalog    # 打印 dataos_operator 算子目录
```

## 校验

```bash
uv run ruff check .
```

## 更多文档

- 实验落盘结构:[`docs/experiment_layout_zh.md`](docs/experiment_layout_zh.md)
- 算子(外部):[`docs/operators_zh.md`](docs/operators_zh.md)
