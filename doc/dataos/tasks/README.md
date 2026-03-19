# DataOS 任务文档索引

本目录把 SFT MVP 的任务按阶段保留，同时补充了当前这轮“先收口、再并行”的执行方式。

## 当前协作约定（2026-03）

- `dataos/` 相关实现此前是多线程并行推进的，没有基于 `git worktree` 做隔离。
- 当前仓库里的 `dataos/` 增量尚未形成稳定的分叉基线，因此不要直接从当前脏工作区继续开多个功能 worktree。
- 先在主工作区完成 `T0 基线收口与重新验证`，形成一个可复用的基线分支或提交，再按任务包开 worktree。

## 文件说明

- `00-roadmap.md`
  - 说明全局路线图、当前真实进度、重排后的任务包、依赖关系和 DoD。
- `01-phase-0-foundation.md`
  - 冻结核心抽象、目录结构和模块边界。
- `02-phase-1-closed-loop.md`
  - 打通 SFT 最小单轮闭环，并标出 `T1` / `T4` 的职责划分。
- `03-phase-2-optimization-validation.md`
  - 补齐优化、恢复、样例和测试，并标出 `T2` / `T3` / `T4` / `T5` 的职责划分。
- `04-phase-3-agentic-upgrades.md`
  - 为后续 agentic 化预留升级接口，并标出 `T3` / `T5` 的职责划分。

## 当前建议任务包

- `T0 基线收口与重新验证`
  - 在当前主工作区执行，不开新 worktree。
- `T1 闭环算子包`
  - 单开一个 worktree，负责基础 SFT 算子和最小闭环里的 operator/eval 落地。
- `T2 Runtime 恢复包`
  - 单开一个 worktree，负责 lineage、checkpoint、retry 和 runtime 恢复能力。
- `T3 Controller 分析包`
  - 单开一个 worktree，负责 controller、result analyzer 和策略接线。
- `T4 示例与验收包`
  - 单开一个 worktree，负责样例数据、任务配置、运行脚本和 E2E。
- `T5 Agentic 扩展包`
  - 单开一个 worktree，负责 rewrite/synthesis/mix/cost 等扩展能力。

## 推荐使用方式

1. 先阅读 `00-roadmap.md` 中的“当前重排原则”和“重排后的任务包”。
2. 在主工作区完成 `T0`，把共享合同和命名约定先冻结。
3. 再按功能域阅读对应 phase 文档，并从已经收口的基线开 worktree。

## 任务记录建议

建议继续在每个任务前维护简单状态：

- `TODO`
- `IN_PROGRESS`
- `DONE`
- `BLOCKED`

这样后面无论是单人推进还是多人协作，都会更清楚。
