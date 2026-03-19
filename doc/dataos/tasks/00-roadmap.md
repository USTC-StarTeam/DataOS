# SFT MVP 路线图

## 1. 路线图目标

目标不是一次性实现完整 Agentic DataOS，而是以最小但完整的闭环方式，逐步建立：

- 声明式任务表达
- 可插拔算子机制
- 顺序执行 runtime
- 评测反馈驱动优化
- 可追踪、可恢复、可复现的工程底座

## 当前进度

- `Phase 0`：`7 / 7` 已完成
- `Phase 1`：`5 / 10` 已完成
- `Phase 2`：`1 / 13` 已完成，`5 / 13` 进行中
- `Phase 3`：`1 / 6` 已完成，`2 / 6` 进行中
- 总计：`14 / 36` 已完成，`7 / 36` 进行中
- 统计口径说明：以上进度按当前代码、测试和边界对象复核，不再沿用最初文档中的预估状态

## 2. 当前重排原则

当前这轮实现有一个现实前提：

- `dataos/` 相关代码之前由多个线程并行推进
- 当时没有基于 `git worktree` 做隔离
- 当前仓库里的 `dataos/` 增量尚未形成稳定分叉基线

因此当前推荐执行方式不是直接继续并行加功能，而是先做一次跨阶段收口：

1. 先在主工作区完成 `T0 基线收口与重新验证`
2. 形成可复用的基线分支或提交
3. 再从这个基线按任务包开独立 worktree

补充说明：

- phase 文档继续负责回答“能力建设属于哪一阶段”
- 任务包负责回答“现在怎么切分、怎么并行、哪些文件要避免互相踩”

## 3. `T0` 基线收口与重新验证

状态：

- `DONE`

目标：

- 收拢当前 `dataos/` 增量，形成稳定基线
- 在功能开发之前先冻结共享合同

重点验证：

- `operator_id` 命名是否统一
  - 当前代码里存在 `filter_by_rule` 和 `cleaning.filter_by_rule` 两种风格，需要先统一
- `simple_sft_eval` 的角色是否明确
  - 需要决定它是 runtime operator、evaluator，还是 operator 调 evaluator 的 adapter
- `planner / runtime / eval / controller` 的边界是否一致
- 文档状态与代码状态是否一致
- 当前非集成测试是否保持通过

完成标准：

- 能从当前工作区切出一个稳定的 `dataos` 基线分支或提交
- 共享命名和边界合同已经冻结
- 任务文档已按真实进度更新
- 后续 worktree 不需要再改同一批共享合同文件

当前结论：

- 内置 SFT runtime operator 统一使用裸 `snake_case`
- `simple_sft_eval` 已冻结为 `evaluator`，不再作为 runtime operator 注入 pipeline
- `planner` 只输出 runtime-only pipeline，`eval` 在 runtime 之后执行
- 共享合同汇总见 `doc/dataos/t0-baseline-contract.md`

## 4. 重排后的任务包

### `T1` 闭环算子包

目标：

- 补齐最小闭环里真正执行数据处理的基础算子

建议任务范围：

- `1.6 filter_by_rule`
- `1.7 dedup`
- `1.8 filter_by_score`
- `1.9 simple_sft_eval` evaluator
- `2.6 sample_by_strategy`

建议 worktree：

- `单开一个 worktree`

建议写入范围：

- `dataos/operators/**`
- `dataos/eval/simple_sft_eval.py`
- 相关 operator / planner / eval 测试

依赖：

- 依赖 `T0`

### `T2` Runtime 恢复包

目标：

- 补齐可恢复运行所需的 runtime 与 storage 能力

建议任务范围：

- `2.1 LineageStore`
- `2.2 CheckpointManager`
- `2.3 RetryPolicy`

建议 worktree：

- `单开一个 worktree`

建议写入范围：

- `dataos/runtime/**`
- `dataos/storage/**`
- 相关 runtime / storage 测试

依赖：

- 依赖 `T0`

### `T3` Controller 分析包

目标：

- 补齐基于评测结果继续迭代的控制层

建议任务范围：

- `2.4 IterativeController`
- `2.5 ResultAnalyzer`
- `3.4 StrategyRegistry` 的实际接线
- `3.6 Planner / Controller` 的升级口接线

建议 worktree：

- `单开一个 worktree`

建议写入范围：

- `dataos/controller/**`
- `dataos/eval/result_analyzer.py`
- `dataos/registry/strategy_registry.py`
- 相关 controller / analyzer / strategy 测试

依赖：

- 依赖 `T0`
- 最好等 `T1` 明确 `EvalResult` 产出格式后再推进

### `T4` 示例与验收包

目标：

- 把最小 MVP 从“模块可测”推进到“样例可跑、E2E 可验”

建议任务范围：

- `1.10` 最小单轮闭环
- `2.8` 最小候选数据集
- `2.9` `TaskSpec` 示例配置
- `2.10` 最小运行脚本
- `2.13` E2E MVP 集成测试

建议 worktree：

- `单开一个 worktree`

建议写入范围：

- `dataos/examples/**`
- `dataos/experiments/**`
- 集成测试和示例文档

依赖：

- 依赖 `T1`
- 最好等 `T2` / `T3` 的主链路稳定后再做最终集成

### `T5` Agentic 扩展包

目标：

- 在不阻塞 MVP 主闭环的前提下，补齐后续 agentic 升级需要的扩展点

建议任务范围：

- `2.7 mix_datasets`
- `3.2 rewrite_with_llm`
- `3.3 synthesize_from_seed`
- `3.5` 成本与反馈建模

建议 worktree：

- `单开一个 worktree`

建议写入范围：

- `dataos/llm/**`
- `dataos/operators/rewriting/**`
- `dataos/operators/synthesis/**`
- 相关 agentic 扩展测试

依赖：

- 依赖 `T0`
- 不阻塞 MVP 最小闭环

## 5. 分阶段规划

### Phase 0：抽象与边界冻结

核心目标：

- 冻结核心 schema
- 冻结目录结构
- 冻结模块职责边界

阶段产出：

- `dataos/` 包骨架
- 核心类与接口定义
- 枚举、schema、抽象对象

详细任务：

- `01-phase-0-foundation.md`

### Phase 1：单轮闭环跑通

核心目标：

- 用最小算子集跑通一次 SFT 数据加工与评测闭环

阶段产出：

- 数据版本管理
- operator registry
- 顺序 runtime
- 规则版 planner
- 基础 SFT 算子
- 简单 eval

详细任务：

- `02-phase-1-closed-loop.md`

### Phase 2：优化、恢复与验证补齐

核心目标：

- 让 MVP 从“能跑”升级到“能迭代、能恢复、能验证”

阶段产出：

- 有限轮 controller
- checkpoint / resume / retry
- lineage 持久化
- 最小可运行样例
- 单元测试和 E2E 测试

详细任务：

- `03-phase-2-optimization-validation.md`

### Phase 3：Agentic 升级接口

核心目标：

- 为后续 LLM/Agent 驱动的数据优化留好接口

阶段产出：

- 统一 LLM abstraction
- strategy registry
- rewrite / synthesis 的 provider 抽象
- 成本与反馈接口增强

详细任务：

- `04-phase-3-agentic-upgrades.md`

## 6. 关键依赖关系

- `TaskSpec`、`DatasetVersion`、`Operator` 是全系统基础，必须先完成。
- `OperatorRegistry` 依赖 `Operator` 抽象。
- `Runtime Executor` 依赖 `OperatorRegistry` 和 `Storage`。
- `Planner` 依赖 `TaskSpec` 与 `OperatorRegistry`。
- `Controller` 依赖 `Planner`、`Runtime`、`Eval`。
- E2E 样例依赖前面所有骨架模块至少具备最小实现。
- `T0` 是所有后续任务包的前置依赖。
- `T1` 和 `T2` 可以在 `T0` 之后并行。
- `T3` 依赖 `T0`，并建议等 `T1` 冻结 eval 合同后推进。
- `T4` 依赖 `T1`，并建议等 `T2` / `T3` 稳定后再做最终集成。
- `T5` 只依赖 `T0`，不阻塞 MVP 主闭环。

## 7. 实施优先级

优先级从高到低建议如下：

1. `T0 基线收口与重新验证`
2. `T1 闭环算子包` 和 `T2 Runtime 恢复包`
3. `T3 Controller 分析包`
4. `T4 示例与验收包`
5. `T5 Agentic 扩展包`

## 8. 每阶段完成标准

### Phase 0 完成标准

- 核心 schema 定义完成
- 模块职责边界清晰
- 目录结构冻结

### Phase 1 完成标准

- 可基于一个 `TaskSpec` 生成并执行最小 pipeline
- 能产出新数据版本和结构化 eval 结果

### Phase 2 完成标准

- 至少支持单轮或有限轮优化
- 支持 checkpoint / resume / retry
- 有最小可运行样例和 E2E 测试

### Phase 3 完成标准

- rewrite / synthesis 不依赖单一 provider
- 上层已能接入更智能的 planner 或 controller

## 9. 总体 Definition of Done

满足以下条件即可认为 SFT MVP 完成：

- 能声明一个完整的 `TaskSpec`
- 能读取候选数据并生成新版本数据集
- 至少执行三类以上 SFT 算子
- 能产出结构化 `EvalResult`
- 能记录 lineage、metrics、artifacts 与 run state
- 能在单轮或有限轮内比较候选配置并给出最佳版本
- 样例脚本和 E2E 测试都能跑通
