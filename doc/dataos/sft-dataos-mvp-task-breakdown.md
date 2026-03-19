# Agentic DataOS SFT MVP 任务拆解总览

这份文档保留任务总览，并补充当前这轮重新收口后的执行顺序。详细拆解仍然按阶段放在 `doc/dataos/tasks/` 目录中。

## 阅读顺序

1. `tasks/README.md`
2. `tasks/00-roadmap.md`
3. `tasks/01-phase-0-foundation.md`
4. `tasks/02-phase-1-closed-loop.md`
5. `tasks/03-phase-2-optimization-validation.md`
6. `tasks/04-phase-3-agentic-upgrades.md`

## 拆分原则

- 总览文件只回答“先做什么、后做什么、为什么这样排”
- 分阶段文件回答“这一阶段要做哪些具体任务、依赖什么、怎么验收”
- 每个任务尽量只解决一个明确问题
- 每个阶段都尽量能形成可验证的增量产出

## 当前推荐执行顺序

当前完成度：

- `Phase 0`：已完成
- `Phase 1`：已完成基础闭环骨架，尚未补齐基础算子与单轮集成
- `Phase 2`：已完成核心 schema 测试，恢复与优化大多停留在接口层
- `Phase 3`：已完成统一 LLM 抽象，策略与 agentic 升级口停留在部分接口层

当前按代码与测试复核后的进度：

- `Phase 0`：`7 / 7` 已完成
- `Phase 1`：`5 / 10` 已完成
- `Phase 2`：`1 / 13` 已完成，`5 / 13` 进行中
- `Phase 3`：`1 / 6` 已完成，`2 / 6` 进行中

## 当前先做什么

由于 `dataos/` 是在没有 `worktree` 隔离的情况下由多个线程并行推进出来的，当前推荐顺序不再是简单地顺着 `Phase 0 -> 1 -> 2 -> 3` 线性做，而是先完成一次跨阶段收口。

### Step 0：先做 `T0 基线收口与重新验证`（已完成）

目标：

- 形成一个可复用的 `dataos` 基线分支或提交
- 统一共享合同，避免后续 worktree 再次冲突

重点验证：

- `operator_id` 命名是否统一
- `simple_sft_eval` 到底是 runtime operator、evaluator，还是两者之间的 adapter
- `planner / runtime / eval / controller` 的公共导出面是否已经稳定
- 文档中的任务状态是否和当前代码真实状态一致

当前结论：

- 内置 SFT runtime operator 统一使用裸 `snake_case`，不再把 `cleaning.filter_by_rule` 这类写法当作共享合同
- `simple_sft_eval` 已冻结为 `evaluator`，不再出现在 runtime pipeline 中
- `PlanningResult.pipeline` 只保留 runtime step，评测 handoff 留在 `PlanningResult.rationale` 和 `TaskSpec.eval_spec`
- 共享合同汇总见 `t0-baseline-contract.md`

### Step 1：并行启动 `T1` 和 `T2`

- `T1 闭环算子包`
  - 负责 `filter_by_rule`、`dedup`、`filter_by_score`、`sample_by_strategy` 和 `simple_sft_eval evaluator`
- `T2 Runtime 恢复包`
  - 负责 `LineageStore`、`CheckpointManager`、`RetryPolicy` 及 runtime 接线

### Step 2：启动 `T3`

- `T3 Controller 分析包`
  - 负责 `IterativeController`、`ResultAnalyzer`、`StrategyRegistry` 的实际接线

### Step 3：启动 `T4`

- `T4 示例与验收包`
  - 负责样例数据、任务配置、运行脚本和 E2E
- 建议在 `T1` 合入后再开，避免做集成时反复跟随 operator 变更

### Step 4：最后启动 `T5`

- `T5 Agentic 扩展包`
  - 负责 `mix_datasets`、`rewrite_with_llm`、`synthesize_from_seed`、成本与反馈建模
- 不阻塞 MVP 主闭环

### Phase 0：抽象与边界冻结

目标：

- 先冻结核心抽象
- 明确目录结构和模块边界
- 避免后续实现过程中反复返工

详细任务：

- `tasks/01-phase-0-foundation.md`

### Phase 1：单轮闭环跑通

目标：

- 在 SFT 场景下跑通最小数据闭环
- 打通 TaskSpec -> Planner -> Runtime -> Eval -> Storage

详细任务：

- `tasks/02-phase-1-closed-loop.md`

执行说明：

- 功能上仍然归属于 Phase 1
- 但实际推进拆成 `T1 闭环算子包` 和 `T4 示例与验收包`

### Phase 2：优化、恢复与样例补齐

目标：

- 补齐多轮优化、失败恢复、样例工程和测试
- 让 MVP 具备更稳定的工程可用性

详细任务：

- `tasks/03-phase-2-optimization-validation.md`

### Phase 3：Agentic 升级接口

目标：

- 在不破坏 MVP 骨架的前提下，为 LLM/Agent 驱动优化预留扩展口

详细任务：

- `tasks/04-phase-3-agentic-upgrades.md`

## 最终完成标准

当以下条件全部满足时，SFT MVP 可以视为完成：

- 能声明一个完整的 `TaskSpec`
- 能读取候选数据并生成新版本数据集
- 至少执行三类以上 SFT 算子
- 能产出结构化 `EvalResult`
- 能记录 lineage、metrics、artifacts 与 run state
- 能在单轮或有限轮内比较候选配置并给出最佳版本
- 样例脚本和 E2E 测试都能跑通
