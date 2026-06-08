---
name: run-dataos
description: >-
  用 DataOS 在 Python 里端到端构建 SFT 训练数据、跑或对比微调实验:用 dataos.load +
  dataos-operator 加载并组装算子成训练 Dataset,再在 dataos.study 容器下跑实验(set_plan
  -> experiment(spec) -> data -> train -> eval -> report -> compare -> review)。你(Claude)是
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
  for spec: exp = study.experiment(spec)           # 据 spec 造 exp.data / exp.train / exp.eval 三个 Stage
    exp.data.run() -> exp.train.run() -> exp.eval.run() -> exp.summary()  # 每个出 report.json
study.summary()  ->  study.review(conclusion)      # 确定性汇总 -> review.json
```

## DataOS 跑在哪(本机 vs 远程)

正式场景:**Claude Code 在用户本机,DataOS 在远程 GPU 服务器。** 本机只有这个 skill(说明书 + 脚本)
和 SSH;dataos 包、`train` / `eval` extra、数据、GPU 都在服务器上。**所以下文所有 `python ...` /
`exp.train(...)` 等,在正式场景里都是你通过 ssh 在服务器上执行,产物(report.json / review.json)读回本机看
——本机无需装 dataos。** 远程怎么装环境、经 ssh 跑实验、读回产物,以及那几个真实坑(PATH / HF 镜像 /
cwd / 代理):见 `references/remote.md`,**真实训练 / 评测前先读它**。

本机离线快验(可选):本机若也装了核心依赖,可只跑 **stub**(`train {"mode":"stub"}`、
`eval {"provider":"stub",...}`,无 GPU / extras)把流水线逻辑过一遍;真实训练 / 评测一律回服务器。

## 第一步:自检环境

在 **dataos 所在的机器上**(正式场景 = 远程服务器,经 ssh)跑一次 `scripts/doctor.py`(脚本在本 skill 的
`scripts/` 下;服务器上即仓库里的 `run-dataos/scripts/doctor.py`)。它确认你 import 的是对的 `dataos`
(cwd 里的 `./dataos` 会盖掉安装的包)、核心依赖和 `train` / `eval` extra 在不在、`llamafactory-cli` 在不在
PATH、GPU 可不可见——第一次跑最容易踩的坑。远程一条命令见 `references/remote.md`。

## 五个阶段(都是你自己在 Python 里调)

只在走到那一阶段时再读对应的 reference——其余别占上下文。

| 阶段 | 你做什么 | 细节 |
|---|---|---|
| **计划** | 写 `plan = {goal, experiments:[spec,...]}`;每个 `spec` 是 `{name, data, train, eval}`。用 `dataos.catalog()` 发现算子。可 copy `assets/plan.template.json` 起步。 | — |
| **数据** | `exp.data` 是 **DataStage**。声明式:`spec["data"]={source,split,limit,field_map,recipe:[{op,params}]}` 再 `exp.data.run()`;或逃生口 `exp.data.set(ds).run()` 注入你用 `dataos_operator` 算子现搭的 `Dataset`。判断全在这。 | `references/data-prep.md` |
| **训练** | `exp.train.run()` -> `Model`,自动取上游 `exp.data` 的产物。开放-config:任意 LLaMA-Factory 键透传;`config` 取 `spec["train"]`。 | `references/train-eval.md` |
| **评估** | `exp.eval.run()` -> `Eval`,自动取上游 `exp.train` 的产物 + `spec["eval"]`。stub 必须显式给。 | `references/train-eval.md` |
| **复盘** | `exp.summary()` -> 单实验 `report.json`(扁平键 + `stages` 段);`study.summary()` -> 汇总/排名;`study.review(conclusion)` -> `review.json`。DataOS 列表,**你**下结论。 | — |

## 容器

- **`dataos.study(id=None, workspace="workspace")` -> `Study`** —— 创建或**重开**(省略 `id`
  则自动流水号 `study-0001`)。方法:`set_plan(plan)`(写 `plan.json`)、`experiment(spec)`、
  `summary()`、`review(conclusion)`。
- **`Study.experiment(spec)` / `dataos.experiment(spec=None)` -> `Experiment`** —— 一个瘦容器,
  据 spec 组合三个 Stage:`exp.data` / `exp.train` / `exp.eval`,外加 `exp.summary()`。每个 spec 给一个
  **唯一的短 `name`**:它是目录 slug,也是 `summary()` 的 join key。

**Stage 自带状态、产物是一等公民。** 每个 stage 记录 `status` / 起止时间 / 耗时 / `error` / `inputs`(血缘)/
产物,落进 `<exp>/{data,train,eval}.json`(= `stage.summary()`)。产物是 **Artifact**(`Data` / `Model` /
`Eval`),带 `id` / `path` / `meta` / `inputs`(上游 artifact id,连成血缘链 **data → model → eval**)。
`exp.summary()` / `study.review()` 就是把下层这些 `summary()` 拼起来——**给 stage 或 artifact 加个字段,会自动往上冒到报告里**。

落盘:`<workspace>/<study_id>/{plan.json, review.json,
experiments/<name>/{data,train,eval}.json + report.json + dataset/ model/ eval/ logs/}`。

## 执行与并行

串行还是铺到多卡并行,是你定的——SDK 两种方式完全一样(study 是目录;`summary()` 读盘上的
`report.json`)。要并行就把数据集准备好一次,然后每个实验用 `scripts/run_one.py`(钉
`CUDA_VISIBLE_DEVICES`)。见 `references/execution.md`。正式场景里这些都在**远程服务器上经 ssh 跑**
(本机 Claude Code 驱动)——见 `references/remote.md`。

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
    exp.data.set(ds).run()             # 逃生口:注入现成 Dataset(声明式则用 spec["data"] + exp.data.run())
    exp.train.run(); exp.eval.run(); exp.summary()
digest = study.summary()               # {varied_keys: ["learning_rate"], best: {name: "b", ...}, ...}
study.review(conclusion="b 胜")         # -> review.json
```

## 边界

- **你是 planner;没有内嵌 agent 或 LLM key。** 如果你为了跑流程去找 API key,就是用错了——
  DataOS 不带 agent / prompt / LLM 客户端。
- **没有把各阶段串起来的一把梭 `run()`,并行是你的事。** 每个 stage 有自己的 `exp.<stage>.run()`,但没有替你
  串 data→train→eval 的总 `run()`;DataOS 不拥有编排循环、也没有 GPU 调度器。
- **算子只来自外部 `dataos_operator` 包**(或用 `dataos.catalog()` 发现);数据集是
  `dataos_operator.Dataset`,不是 DataOS 类型。
- **核对真实文件。** 读 `experiments/<name>/report.json` 和 `review.json`;别编造结论。
