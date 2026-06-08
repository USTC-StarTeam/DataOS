# 落盘目录结构

DataOS 把一次研究的所有产物落在一个 **project 目录**下(project ⊃ study ⊃ experiment)。**都是目录**,
`summary()` 直接读盘聚合,所以无论串行还是多进程并行,布局都一样。

## 三层结构

```text
<workspace>/<project_id>/
  project.json                      # 项目元信息(goal / 初始 instruct / 时间戳)
  <study_id>/                       # 一轮 Study(一个 plan)
    plan.json                       # 多实验计划
    review.json                     # 复盘(summary 聚合 + 你写的 conclusion)
    experiments/
      <name>/                       # 一个 Experiment(name = spec 里的唯一短名,作目录 slug)
        data.json                   # DataStage 自记状态(= summary():status/时间/inputs/产物 Data)
        train.json                  # TrainStage 自记状态(含产物 Model + 血缘 inputs)
        eval.json                   # EvalStage 自记状态(含产物 Eval + 血缘 inputs)
        dataset/                    # 训练集物化(Data artifact 的 path)
        model/                      # 训练产物(LoRA adapter / 全量权重;Model 的 path)
        eval/                       # 评测产物(results.json 等;Eval 的 path)
        logs/                       # 各阶段日志
        _lf/                        # LLaMA-Factory 暂存(train.yaml 等)
        report.json                 # 单实验总结(扁平键 + stages 段)
```

`<project_id>` / `<study_id>` 由 `dataos.project(id=...)` / `project.study(id=...)` 指定或自动分配
(`project-0001` / `study-0001` …)。两者都 **open-or-create**:目录已存在则重开、读回 `project.json` / `plan.json`,
所以不同进程能用同一 id attach、**以 project 为单位 resume**(`project.load()` 把整个项目回灌)。

## report.json(单实验)

由各 stage `run()` **自动写**(每跑完一个 stage 刷新一次);`exp.summary()` 是**纯 getter**,返回当前情况、不写盘。内容由各 stage / artifact 的 `summary()` 拼成。扁平键:`id` / `name` / `spec` / `status`
(由三 stage 状态派生:任一 failed→failed,否则取最远完成阶段)/ `created_at` / `updated_at` /
`dataset`{path, format, rows} / `model`{id, path, base_model, finetuning_type, mode} /
`eval`{id, score, metrics, path};外加 `stages`{data, train, eval} —— 每个是该 stage 的完整 `summary()`
(status / 起止 / 耗时 / error / inputs / config / artifact)。给 stage 或 artifact 加字段会自动出现在
`stages` 段里。`spec` 进 report 是为了让 `summary()` 做配置 diff。

## review.json(study 级)

`study.review(conclusion=...)` 写出,= `summary()` 的确定性汇总 + 你写的 `conclusion`。字段:
`study_id` / `goal` / `n_experiments` / `varied_keys` / `experiments[]` / `ranking[]` / `best` /
`metric_winners` / `caution`(headline 冠军与某指标冠军不一致时的提示)/ `conclusion` / `status` / 时间戳。
`summary()` 只 glob `experiments/*/report.json`,容忍还在跑或缺失的实验。

## workspace 的意义

`workspace` 只是 project 根的上层目录(默认 `workspace`)。同一个 `workspace` 下可有多个 project,每个
project 下多个 study:

```text
workspace/
  project-0001/
    project.json
    study-0001/...
    lr-sweep/...
```
