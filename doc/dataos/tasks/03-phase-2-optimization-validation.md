# Phase 2：优化、恢复与验证补齐

## 阶段目标

这一阶段把 MVP 从“能跑一次”升级为“能多轮迭代、能失败恢复、能稳定验证”。

当前进度：

- `1 / 13` 已完成
- `5 / 13` 进行中

补充说明：

- 本阶段有不少任务已经落到了“边界对象或协议已存在，但具体实现还没接上”的状态。
- 这一阶段现在拆成多个任务包推进：
  - `T2 Runtime 恢复包`：`2.1` 到 `2.3`
  - `T3 Controller 分析包`：`2.4` 到 `2.5`
  - `T1 闭环算子包`：`2.6`
  - `T5 Agentic 扩展包`：`2.7`
  - `T4 示例与验收包`：`2.8` 到 `2.10`、`2.13`
  - `2.12` 作为跨包回归测试持续补齐

## 推荐 worktree 划分

- `T2`：
  - 单开一个 worktree，集中修改 `dataos/runtime/**` 和 `dataos/storage/**`
- `T3`：
  - 单开一个 worktree，集中修改 `dataos/controller/**`、`dataos/eval/result_analyzer.py` 和策略接线
- `T4`：
  - 单开一个 worktree，在 `T1` 合入后再做样例、脚本和 E2E
- `T5`：
  - 单开一个 worktree，低优先级处理 agentic 扩展

## 当前优先验证点

- `LineageStore`、`CheckpointManager`、`RetryPolicy` 现在只有边界还是已经有可用最小实现
- `ResultAnalyzer` 和 `IterativeController` 的输入输出合同是否已够稳定
- 当前测试覆盖是否已经足以保护 `RunStore`、`RuntimeExecutor` 和核心 schema

## Task 2.1：实现 `LineageStore`

状态：

- `IN_PROGRESS`

当前情况：

- `storage/lineage_store.py` 已经有协议边界，但还没有具体实现和接线

归属任务包：

- `T2`

目标：

- 单独沉淀数据版本依赖与变换链路。

建议产出：

- `storage/lineage_store.py`

验收标准：

- 可从最终版本追溯到输入版本与执行步骤

## Task 2.2：实现 `CheckpointManager`

状态：

- `IN_PROGRESS`

当前情况：

- `runtime/checkpoint.py`、`runtime/state_store.py` 已经定义了协议，但还没有完整的 checkpoint / restore 实现

归属任务包：

- `T2`

目标：

- 支持 step 级 checkpoint。

建议产出：

- `runtime/checkpoint.py`
- `runtime/state_store.py`

验收标准：

- 流程中断后可从最近成功 step 恢复

## Task 2.3：实现 `RetryPolicy`

状态：

- `IN_PROGRESS`

当前情况：

- `runtime/retry.py` 已有边界定义，但 runtime 主执行链路还没有真正接入重试逻辑

归属任务包：

- `T2`

目标：

- 为失败的 operator 提供有限重试能力。

建议产出：

- `runtime/retry.py`

验收标准：

- 支持最大重试次数
- 支持基础退避策略

## Task 2.4：实现 `IterativeController`

状态：

- `TODO`

归属任务包：

- `T3`

目标：

- 管理有限轮优化和最佳结果选择。

建议产出：

- `controller/iterative_controller.py`

最低要求：

- 维护 round 状态
- 比较候选配置
- 判断是否停止

验收标准：

- 能进行至少 1 到 3 轮简单迭代

## Task 2.5：实现 `ResultAnalyzer`

状态：

- `IN_PROGRESS`

当前情况：

- `eval/result_analyzer.py` 已经有 `AnalysisSummary` 和 `ResultAnalyzer` 协议，但还缺少具体规则分析实现

归属任务包：

- `T3`

目标：

- 根据 eval 结果给出规则版调参建议。

建议产出：

- `eval/result_analyzer.py`

第一版建议：

- 调整 score threshold
- 调整 sample ratio
- 调整 source mix ratio

验收标准：

- 能把评测结果映射为新的候选配置

## Task 2.6：实现 `sample_by_strategy`

状态：

- `TODO`

归属任务包：

- `T1`

目标：

- 支持按来源、标签、难度桶等采样。

验收标准：

- 支持固定随机种子
- 输出采样前后分布统计

## Task 2.7：实现 `mix_datasets`

状态：

- `TODO`

归属任务包：

- `T5`

目标：

- 支持多个数据版本按比例混合。

验收标准：

- 支持显式配比
- 输出混合后组成统计

## Task 2.8：准备最小候选数据集

状态：

- `TODO`

归属任务包：

- `T4`

目标：

- 提供一个稳定可复现的小样例输入。

建议产出：

- `examples/sft_candidate.jsonl`

样例应包含：

- 重复样本
- 低分样本
- 可过滤样本
- 不同来源样本

验收标准：

- 能触发多个算子和评测指标变化

## Task 2.9：编写 `TaskSpec` 示例配置

状态：

- `TODO`

归属任务包：

- `T4`

目标：

- 提供完整的 SFT MVP 任务配置示例。

建议产出：

- `experiments/sft/tasks/*.yaml`
- 或 `experiments/sft/tasks/*.json`

验收标准：

- 能声明输入数据、约束、预算、评测配置和 allowed operators

## Task 2.10：编写最小运行脚本

状态：

- `BLOCKED`

归属任务包：

- `T4`

目标：

- 提供一条命令跑通 MVP 的实验入口。

建议产出：

- `experiments/sft/run_sft_mvp.py`

验收标准：

- 输入候选数据
- 输出新版本数据、metrics、eval 与 lineage

## Task 2.11：补齐核心 schema 测试

状态：

- `DONE`

当前情况：

- 当前已经有覆盖 `TaskSpec`、`DatasetVersion`、`PipelineStep`、`EvalResult`、`OptimizationState` 的回归测试

覆盖对象：

- `TaskSpec`
- `DatasetVersion`
- `PipelineStep`
- `EvalResult`
- `OptimizationState`

验收标准：

- 核心对象序列化、默认值和约束行为可验证

## Task 2.12：补齐 storage / runtime / operator 测试

状态：

- `IN_PROGRESS`

当前情况：

- 已有 `DatasetStore`、`RunStore`、`RuntimeExecutor`、核心 operator contract 的测试
- 仍需继续补齐 lineage / checkpoint / retry / 基础算子 / E2E 前置覆盖

覆盖范围：

- 数据版本落盘与读取
- lineage 记录
- run 状态恢复
- 多 step 顺序执行
- checkpoint / retry
- 基础算子行为

验收标准：

- 核心底座具备稳定的回归保护

## Task 2.13：补齐 E2E MVP 集成测试

状态：

- `BLOCKED`

归属任务包：

- `T4`

目标：

- 一次测试验证完整闭环。

覆盖范围：

- 候选数据读取
- pipeline 执行
- 评测结果生成

## Phase 2 完成定义

以下条件全部满足后，Phase 2 才算完成：

- 至少支持单轮或有限轮优化
- 支持 checkpoint / resume / retry
- 有最小可运行样例和 E2E 测试
- storage / runtime / controller 的关键回归测试已经补齐
- lineage 和 metrics 落盘

验收标准：

- 一次运行可验证从输入到输出的完整链路

## Phase 2 完成定义

以下条件全部满足后，Phase 2 才算完成：

- 支持有限轮优化
- 支持 checkpoint / resume / retry
- 有最小可运行样例
- 有单元测试和 E2E 测试
