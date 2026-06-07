# 落盘目录结构

DataOS 把一次研究的所有产物落在一个 study 目录下。**study 就是一个目录**,`compare()` 直接读盘聚合,
所以无论串行还是多进程并行,布局都一样。

## 两层结构

```text
<workspace>/<study_id>/
  plan.json                       # 多实验计划
  review.json                     # 复盘(compare 聚合 + 你写的 conclusion)
  experiments/
    <name>/                       # 一个 Experiment(name = spec 里的唯一短名,作目录 slug)
      dataset/                    # 训练集物化(真实训练时)
      model/                      # 训练产物(LoRA adapter / 全量权重)
      eval/                       # 评测产物(results.json 等)
      logs/                       # 各阶段日志
      _lf/                        # LLaMA-Factory 暂存(train.yaml 等)
      report.json                 # 单实验总结
```

`<study_id>` 由 `dataos.study(id=...)` 指定或自动分配(`study-0001` …)。`dataos.study(id=...)` 是
**open-or-create**:目录已存在则重开、读回 `plan.json`,所以不同进程能用同一 id attach 到同一个 study。

## report.json(单实验)

`exp.finalize()` 写出。字段:`id` / `name` / `spec` / `status` / `created_at` / `updated_at` /
`steps` / `dataset`{path, format, rows} / `model`{id, path, base_model, finetuning_type, mode} /
`eval`{id, score, metrics, path}。`spec` 进 report 是为了让 `compare()` 做配置 diff。

## review.json(study 级)

`study.finalize(conclusion=...)` 写出,= `compare()` 的确定性对比 + 你写的 `conclusion`。字段:
`study_id` / `goal` / `n_experiments` / `varied_keys` / `experiments[]` / `ranking[]` / `best` /
`conclusion` / `status` / 时间戳。`compare()` 只 glob `experiments/*/report.json`,容忍还在跑或缺失的实验。

## workspace 的意义

`workspace` 只是 study 根的上层目录(默认 `workspace`)。同一个 `workspace` 下可以有多个 study:

```text
workspace/
  study-0001/...
  lr-sweep/...
```
