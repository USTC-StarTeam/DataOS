---
name: run-dataos
description: >-
  用 DataOS 在 Python 里端到端构建 SFT 训练数据、跑或对比微调实验:用 dataos.load +
  dataos-operator 加载并组装算子成训练 Dataset,再在 dataos.study 容器下跑实验(set_plan
  -> experiment(spec) -> train -> eval -> finalize -> compare -> review)。你(Claude)是
  runtime 和 planner——写 plan、驱动各阶段、决定是否并行;DataOS 不调 LLM、也没有 run()。
  凡用户想微调/训练模型、构建或准备 SFT / Alpaca / ShareGPT 数据集、扫参或对比训练配置
  (learning rate、LoRA vs 全参、数据配方)、用 lm-eval 评测模型、跑无 GPU 的 stub 流程,或提到
  DataOS、dataos-operator、dataos.study / dataos.experiment、LoRA、LLaMA-Factory、产出
  DataOS report / review——哪怕没明说 "DataOS"——都用本 skill。(fine-tune or train a model,
  build an SFT/Alpaca/ShareGPT dataset, sweep or compare training configs, evaluate with
  lm-eval, run a no-GPU stub pipeline.)
---

# 跑 DataOS 实验(skill 驱动)

DataOS 是一组**确定性执行器** + 两个容器——**`Study`**(同一个 plan 下的一组实验)和
**`Experiment`**(单次运行)。**你是 runtime 和 planner。** 没有内嵌 LLM、没有计划 agent、
没有把各阶段串起来的 `run()`、也没有调度器——plan 是你写的,各阶段是你驱动的,要不要并行是你定的。

一个 plan 通常描述**不止一个实验**(扫学习率、比 LoRA / 全参、试数据配方),最后收敛成一份确定性对比:

```
plan = {goal, experiments: [spec, ...]}            # 你写
study = dataos.study(id); study.set_plan(plan)     # 容器(一个目录)
  for spec: exp = study.experiment(spec) -> 数据 -> train -> eval -> finalize   # 每个出 report.json
study.compare()  ->  study.finalize(conclusion)    # 确定性对比表 -> review.json
```

## 第一步:自检环境

先跑一次 `python scripts/doctor.py`。它确认你 import 的是对的 `dataos`(cwd 里的 `./dataos`
会盖掉安装的包)、核心依赖和 `train` / `eval` extra 在不在、`llamafactory-cli` 在不在 PATH、
GPU 可不可见——这些都是第一次跑最容易踩的坑。

## 五个阶段(都是你自己在 Python 里调)

只在走到那一阶段时再读对应的 reference——其余别占上下文。

| 阶段 | 你做什么 | 细节 |
|---|---|---|
| **计划** | 写 `plan = {goal, experiments:[spec,...]}`;每个 `spec` 是 `{name, data, train, eval}`。用 `dataos.catalog()` 发现算子。可 copy `assets/plan.template.json` 起步。 | — |
| **数据** | `dataos.load(...)`,再用 `dataos_operator` 算子整形。判断全在这——处理**没有 DataOS API**,你直接 `Op(**p).run(ds)`。 | `references/data-prep.md` |
| **训练** | `exp.train(ds, config=None)` -> `Model`。开放-config:任意 LLaMA-Factory 键透传;`config` 默认取 `spec["train"]`。 | `references/train-eval.md` |
| **评估** | `exp.eval(model=None, config=None)` -> `Eval`。默认取本实验的模型 + `spec["eval"]`。stub 必须显式给。 | `references/train-eval.md` |
| **复盘** | `exp.finalize()` -> 单实验 `report.json`;`study.compare()` -> 对比;`study.finalize(conclusion)` -> `review.json`。DataOS 列表,**你**下结论。 | — |

## 容器

- **`dataos.study(id=None, workspace="workspace")` -> `Study`** —— 创建或**重开**(省略 `id`
  则自动流水号 `study-0001`)。方法:`set_plan(plan)`(写 `plan.json`)、`experiment(spec)`、
  `compare()`、`finalize(conclusion)`。
- **`Study.experiment(spec)` / `dataos.experiment(spec=None)` -> `Experiment`** ——
  `train` / `eval` / `finalize`。每个 spec 给一个**唯一的短 `name`**:它是目录 slug,也是
  `compare()` 的 join key。

落盘:`<workspace>/<study_id>/{plan.json, review.json, experiments/<name>/.../report.json}`。

## 执行与并行

串行还是铺到多卡并行,是你定的——SDK 两种方式完全一样(study 是目录;`compare()` 读盘上的
`report.json`)。要并行就把数据集准备好一次,然后每个实验用 `scripts/run_one.py`(钉
`CUDA_VISIBLE_DEVICES`)。见 `references/execution.md`。

## 示例(stub,无 GPU / 无 LLM key)

```python
import dataos
from dataos_operator import Dataset

study = dataos.study(id="lr-smoke")
study.set_plan({"goal": "比较两个 stub 配置", "experiments": [
    {"name": "a", "train": {"mode": "stub", "learning_rate": 1e-4},
     "eval": {"provider": "stub", "provider_config": {"score": 0.1}}},
    {"name": "b", "train": {"mode": "stub", "learning_rate": 5e-5},
     "eval": {"provider": "stub", "provider_config": {"score": 0.2}}},
]})
for spec in study.plan["experiments"]:
    exp = study.experiment(spec)
    ds = Dataset.from_records(
        [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(8)],
        metadata={"dataset_format": "alpaca"})
    exp.train(ds); exp.eval(); exp.finalize()
cmp = study.compare()                  # {varied_keys: ["learning_rate"], best: {name: "b", ...}, ...}
study.finalize(conclusion="b 胜")       # -> review.json
```

## 边界

- **你是 planner;没有内嵌 agent 或 LLM key。** 如果你为了跑流程去找 API key,就是用错了——
  DataOS 不带 agent / prompt / LLM 客户端。
- **没有一把梭的 `run()`,并行是你的事。** 各阶段自己调;DataOS 不拥有编排循环、也没有 GPU 调度器。
- **算子只来自外部 `dataos_operator` 包**(或用 `dataos.catalog()` 发现);数据集是
  `dataos_operator.Dataset`,不是 DataOS 类型。
- **核对真实文件。** 读 `experiments/<name>/report.json` 和 `review.json`;别编造结论。
