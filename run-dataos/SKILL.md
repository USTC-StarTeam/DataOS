---
name: run-dataos
description: >-
  用 DataOS 在 Python 里端到端构建 SFT 训练数据、跑或对比微调实验:用 dataos.load +
  dataos-operator 加载并组装算子成训练 Dataset,在一个可恢复的 dataos.project 下按 study ->
  experiment 跑(project.study -> set_plan -> experiment(spec) -> data -> train -> eval ->
  summary -> review)。你(Claude)是 runtime 和 planner——写 plan、驱动各阶段、决定是否并行;
  DataOS 不调 LLM、也没有一把梭 run()。凡用户想微调/训练模型、构建或准备 SFT / Alpaca / ShareGPT
  数据集、扫参或对比训练配置(learning rate、LoRA vs 全参、数据配方)、用 lm-eval 评测模型、跑无 GPU
  的 stub 流程,或提到 DataOS、dataos-operator、dataos.project / study / experiment、LoRA、
  LLaMA-Factory、产出 DataOS report / review——哪怕没明说 "DataOS"——都用本 skill。(fine-tune or
  train a model, build an SFT/Alpaca/ShareGPT dataset, sweep or compare training configs,
  evaluate with lm-eval, run a no-GPU stub pipeline.)
---

# 跑 DataOS 实验(skill 驱动)

DataOS 是一组**确定性执行器** + 三层容器:**`Project`**(一个研究项目 = 一串 study,**resume 的基本单位**)
⊃ **`Study`**(同一个 plan 下的一组实验)⊃ **`Experiment`**(单次运行)。**你是 runtime 和 planner。**
没有内嵌 LLM、没有计划 agent、没有把各阶段串起来的 `run()`、也没有调度器——plan 是你写的,各阶段是你
驱动的,要不要并行是你定的。

层次与数据流:

```
Project  dataos.project(id)              # 可恢复的研究项目(一串 study)
  └─ Study  project.study(id)            # 一轮 = 一个 plan
       set_plan(plan)
       └─ Experiment  study.experiment(spec)
            exp.data.run() -> exp.train.run() -> exp.eval.run()   # 每个 run() 落 {kind}.json + 刷 report.json
            #   data  ->  train  ->  eval  ->  (report.json)
       study.summary() -> study.review(conclusion)                # 确定性汇总 -> review.json
```

一轮 study 通常描述**不止一个实验**(扫学习率、比 LoRA / 全参、试数据配方),收敛成一份 `review.json`。

## 多 study 迭代(在一个 project 内,**你**驱动)

一个 **project** 就是"一个又一个 study 的循环":**首轮 study 的 plan 来自用户初始 instruct;后续每轮 plan 来
自你对上一轮 `review.json` 的判断**。这条外层循环是 runtime(你)的事——DataOS 不生成 plan、也不串 study。

```
用户 instruct ──你写 plan──▶ project.study(1)（多实验[data→train→eval] → review.json）
review.json   ──你读它、定方向、写新 plan──▶ project.study(2)（新 study，新 plan → … → review.json）
review.json   ──▶ project.study(3) ……（重复,直到目标达成)
```

- 用 `proj.set_goal(instruct=...)` 把用户初始诉求记进 `project.json`;每轮 study 用有意义的 `id`
  (`lr-sweep` → `full-vs-lora` → `recipe-v2`),旧轮产物原样留在 project 目录里、可回溯对比。
- **plan 的生成始终是你的判断,不是 DataOS 的函数**——别找"自动生成下一轮 plan"的 API,没有、也不会有。

## 重开与续跑(resume 以 **project** 为单位)

一切落盘,所以**用 Claude 重开一个老项目 = 把整个研究读回来**:

```python
proj = dataos.project(id="qwen-gsm8k")   # 读回 project.json(目标 / instruct)
studies = proj.load()                    # 盘上所有 study + experiment + stage 的状态 / 血缘 / 产物全回灌
proj.summary()                           # 各 study 的 best / varied_keys 总览
```

**再驱动即续跑**:重新跑 plan,`exp.<stage>.run()` 遇到已 `completed` 的 stage 会**跳过**(只补 pending /
failed),`run(force=True)` 才强制重跑。所以"换进程 / 换机 / 隔几天接着跑"都成立——不需要 session / 守护进程。

## DataOS 跑在哪(本机 vs 远程)

正式场景:**Claude Code 在用户本机,DataOS 在远程 GPU 服务器。** 本机只有这个 skill(说明书 + 脚本)
和 SSH;dataos 包、`train` / `eval` extra、数据、GPU 都在服务器上。**所以下文所有 `python ...` /
`exp.train.run()` 等,在正式场景里都是你通过 ssh 在服务器上执行,产物(report.json / review.json)读回本机看
——本机无需装 dataos。** 远程怎么装环境、经 ssh 跑实验、读回产物,以及那几个真实坑(PATH / HF 镜像 /
cwd / 代理):见 `references/remote.md`,**真实训练 / 评测前先读它**。

本机离线快验(可选):本机若也装了核心依赖,可只跑 **stub**(`train {"mode":"stub"}`、
`eval {"provider":"stub",...}`,无 GPU / extras)把流水线逻辑过一遍;真实训练 / 评测一律回服务器。

## 第一步:自检环境

在 **dataos 所在的机器上**(正式场景 = 远程服务器,经 ssh)跑一次 `scripts/doctor.py`(脚本在本 skill 的
`scripts/` 下;服务器上即仓库里的 `run-dataos/scripts/doctor.py`)。它确认你 import 的是对的 `dataos`
(cwd 里的 `./dataos` 会盖掉安装的包)、核心依赖和 `train` / `eval` extra 在不在、`llamafactory-cli` 在不在
PATH、GPU 可不可见——第一次跑最容易踩的坑。远程一条命令见 `references/remote.md`。

## 五个阶段(study 内,你自己在 Python 里逐个调)

只在走到那一阶段时再读对应的 reference——其余别占上下文。

| 阶段 | 你做什么 | 细节 |
|---|---|---|
| **计划** | 建 / 重开 `project`(`proj.set_goal(instruct=...)` 记初始诉求);写 `plan = {goal, experiments:[spec,...]}` 交给 `study.set_plan`,每个 `spec` 是 `{name, data, train, eval}`。用 `dataos.catalog()` 发现算子。可 copy `assets/plan.template.json` 起步。 | — |
| **数据** | `exp.data` 是 **DataStage**。声明式:`spec["data"]={source,split,limit,field_map,recipe:[{op,params}]}` 再 `exp.data.run()`;或逃生口 `exp.data.set(ds).run()` 注入你用 `dataos_operator` 算子现搭的 `Dataset`。判断全在这。 | `references/data-prep.md` |
| **训练** | `exp.train.run()` -> `Model`,自动取上游 `exp.data` 的产物。开放-config:任意 LLaMA-Factory 键透传;`config` 取 `spec["train"]`。 | `references/train-eval.md` |
| **评估** | `exp.eval.run()` -> `Eval`,自动取上游 `exp.train` 的产物 + `spec["eval"]`。stub 必须显式给。 | `references/train-eval.md` |
| **复盘** | `report.json` 由各 stage `run()` **自动落**(扁平键 + `stages` 段);`exp.summary()` 是**纯 getter**,返回该实验当前情况;`study.summary()` -> 汇总/排名;`study.review(conclusion)` -> `review.json`。DataOS 列表,**你**下结论。 | — |

## 容器

- **`dataos.project(id=None, workspace="workspace")` -> `Project`** —— 创建或**重开**研究项目(省略 `id`
  则自动流水号 `project-0001`)。方法:`study(id)`(在项目下建/重开 study)、`studies`、`load()`(重开时把整个
  项目回灌成对象)、`summary()`(各 study 总览)、`set_goal(goal=, instruct=)`(写 `project.json`)。
- **`project.study(id=None)` -> `Study`** —— 在项目下创建或**重开** study(省略 `id` 则 `study-0001`)。方法:
  `set_plan(plan)`(写 `plan.json`)、`experiment(spec)`、`summary()`、`review(conclusion)`、`load_experiments()`。
  **study 只能经 project 建**(没有独立的 `dataos.study`)。
- **`study.experiment(spec)` -> `Experiment`** —— 一个瘦容器,据 spec 组合三个 Stage:`exp.data` / `exp.train` /
  `exp.eval`,外加 `exp.summary()`。**open-or-create**:该 `name` 已跑过(盘上有 `report.json`)就**回灌状态、
  复用 exp_id**。每个 spec 给一个**唯一的短 `name`**:它是目录 slug,也是 `summary()` 的 join key。

**Stage 自带状态、产物是一等公民。** 每个 stage 记录 `status` / 起止时间 / 耗时 / `error` / `inputs`(血缘)/
产物,落进 `<exp>/{data,train,eval}.json`(= `stage.summary()`)。产物是 **Artifact**(`Data` / `Model` /
`Eval`),带 `id` / `path` / `meta` / `inputs`(上游 artifact id,连成血缘链 **data → model → eval**)。
`exp.summary()` / `study.review()` 就是把下层这些 `summary()` 拼起来——**给 stage 或 artifact 加个字段,会自动往上冒到报告里**。

**重开即续跑**:`run()` 遇到已 `completed` 的 stage **跳过**(只补 pending / failed),`run(force=True)` 才重跑。
所以重新驱动一个重开的 project/study 就是接着没跑完的跑。

落盘:`<workspace>/<project_id>/{project.json,
<study_id>/{plan.json, review.json, experiments/<name>/{data,train,eval}.json + report.json + dataset/ model/ eval/ logs/}}`。

## 执行与并行

串行还是铺到多卡并行,是你定的——SDK 两种方式完全一样(都是目录;`summary()` 读盘上的 `report.json`)。要并行就
把数据集准备好一次,然后每个实验用 `scripts/run_one.py`(钉 `CUDA_VISIBLE_DEVICES`,带 `--project --study --spec`)。
见 `references/execution.md`。正式场景里这些都在**远程服务器上经 ssh 跑**(本机 Claude Code 驱动)——见
`references/remote.md`。

**复现(docker 可选)**:每个实验跑完自动落 `manifest.json`(环境版本 + 数据 fingerprint + config + seeds + 硬件);
`run_one.py --replay manifest.json` **直接复现并验证,无需 docker**——`uv.lock`(`uv sync --frozen`)钉 python 层 + manifest
就够了。**docker 只是可选的环境钉死方案**:再把 OS / CUDA 整层也钉死(`assets/Dockerfile` 按 digest + `scripts/run_in_docker.sh`)。
另有 subagent 自愈并行执行模式。详见 `references/execution.md`;DataOS 不感知 docker(这层是运行时的)。

## 示例(stub,无 GPU / 无 LLM key)

```python
import dataos
from dataos_operator import Dataset

proj = dataos.project(id="lr-smoke")              # 研究项目(可恢复)
proj.set_goal(instruct="比较两个 stub 配置")
study = proj.study(id="lr-sweep")                 # 项目下的一轮 study
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
    exp.train.run(); exp.eval.run()    # report.json 自动落;exp.summary() 可随时取当前情况
digest = study.summary()               # {varied_keys: ["learning_rate"], best: {name: "b", ...}, ...}
study.review(conclusion="b 胜")         # -> review.json
# 重开续跑:dataos.project(id="lr-smoke").load();再驱动,completed 的自动跳过
```

## 边界

- **你是 planner;没有内嵌 agent 或 LLM key。** 如果你为了跑流程去找 API key,就是用错了——
  DataOS 不带 agent / prompt / LLM 客户端。
- **没有把各阶段串起来的一把梭 `run()`,并行是你的事。** 每个 stage 有自己的 `exp.<stage>.run()`,但没有替你
  串 data→train→eval 的总 `run()`;DataOS 不拥有编排循环、也没有 GPU 调度器。
- **study / experiment 都在 project 之下**:`project.study(id)` 建 study,`study.experiment(spec)` 建实验;
  没有独立的 `dataos.study` / `dataos.experiment`。
- **算子只来自外部 `dataos_operator` 包**(或用 `dataos.catalog()` 发现);数据集是
  `dataos_operator.Dataset`,不是 DataOS 类型。
- **核对真实文件。** 读 `project.json` / `<study>/review.json` / `experiments/<name>/report.json`;别编造结论。
