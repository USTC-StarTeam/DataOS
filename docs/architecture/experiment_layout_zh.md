# 实验目录结构说明

本文说明 DataOS 当前一次实验的唯一落盘目录结构。

## 设计目标

- 同一个实验只保留一个根目录
- 输入数据、运行产物、日志和最终报告都放在同一个实验目录下
- 不再把实验内容复制到另一套全局归档目录里
- `report.json` 是这次实验的唯一主索引

## 默认目录

如果运行命令里只传 `--workspace workspace`，不额外传 `--sandbox`，那么每次实验默认会创建：

```text
workspace/<exp_id>/
```

其中 `<exp_id>` 是运行时自动生成的唯一实验 ID，所以同一个 `workspace` 连续跑多次也不会冲突。

## 目录结构

一个典型实验目录会长成这样：

```text
workspace/
  exp-20260512.../
      inputs/
        dataset-.../
      artifacts/
        datasets/
          dataset-.../
        models/
          model-.../
        evals/
          eval-.../
        results/
          result-.../
      logs/
      tmp/
      code/
      report.json
      meta.json
```

## 各部分含义

- `inputs/`
  - 本次实验最初读入的数据集
  - 每个输入数据集会有自己的 `dataset-...` 目录

- `artifacts/`
  - 本次实验过程中真正产生的新工件
  - 按类型分为 `datasets/`、`models/`、`evals/`、`results/`

- `logs/`
  - 算子日志和运行阶段日志

- `tmp/`
  - 临时文件

- `code/`
  - 运行时需要保留的辅助代码或脚本内容

- `report.json`
  - 这次实验的唯一主记录
  - 包含 task、steps、artifacts、status、score、plan、review 等内容

- `meta.json`
  - 实验根目录对应的 `ExperimentArtifact` 元数据

## report.json 保存什么

`report.json` 现在统一收口保存：

- `exp_id`
- `task`
- `experiment_root`
- `artifacts`
- `steps`
- `status`
- `score`
- `error`
- `created_at`
- `started_at`
- `finished_at`
- `plan`
- `review`

也就是说，过去分散保存的 `plan.json`、`review.json` 的核心内容，现在都并入 `report.json` 里，不再额外重复落盘。

## 为什么这样更简单

旧方案里常见的问题是：

- 运行现场在一套目录里
- 归档副本又在另一套全局目录里
- `report.json` 还会在多个位置重复出现

新方案改成单实验单目录之后：

- 看一次实验只需要进入一个目录
- 不需要在两套目录之间来回跳
- 不会重复复制整份 experiment artifact
- 数据工件和报告更容易统一管理

## 和 `--workspace` 的关系

`--workspace` 只是实验根集合的上层目录，不是单次实验目录本身。

例如：

```bash
uv run dataos-auto run \
  --project gsm8k-smoke \
  --target-bench gsm8k \
  --ptb-profile smoke \
  --runtime-config configs/runtimes/h100.json
```

这条命令多跑几次时，会得到：

```text
workspace/exp-a.../
workspace/exp-b.../
workspace/exp-c.../
```

所以共享同一个 `workspace` 没问题，真正区分不同实验的是自动生成的 `exp_id`。
