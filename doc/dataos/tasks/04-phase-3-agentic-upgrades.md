# Phase 3：Agentic 升级接口

## 阶段目标

这一阶段不是立即把系统做成复杂 agent，而是在不破坏 MVP 的前提下，把几个关键扩展点设计好。

当前进度：

- `1 / 6` 已完成
- `2 / 6` 进行中

补充说明：

- 当前 Phase 3 不阻塞 MVP 主闭环。
- 这一阶段现在拆成两个任务包：
  - `T3 Controller 分析包`：负责 `3.4`、`3.6`
  - `T5 Agentic 扩展包`：负责 `3.2`、`3.3`、`3.5`

## 推荐 worktree 划分

- `T3`：
  - 单开一个 worktree，负责策略注册和 Planner / Controller 升级口接线
- `T5`：
  - 单开一个 worktree，负责 rewrite / synthesis / cost 扩展

## 当前优先验证点

- `StrategyRegistry` 是否只是存在于代码里，还是已经被 planner / controller 真实消费
- 当前 `Planner` / `Controller` / `Runtime` 的边界是否足够支持后续更智能策略替换
- `rewrite_with_llm` 和 `synthesize_from_seed` 是否应在 MVP 完成后再接入

## Task 3.1：抽象统一 LLM 接口

状态：

- `DONE`

目标：

- 让重写、合成、评测增强能力都不绑定单一 provider。

建议产出：

- `llm/base.py`
- `llm/adapters/openai.py`
- `llm/adapters/vllm.py`

验收标准：

- 上层算子只依赖统一接口
- provider 可以替换

## Task 3.2：实现 `rewrite_with_llm`

状态：

- `TODO`

归属任务包：

- `T5`

目标：

- 把重写能力接入统一 LLM 抽象。

验收标准：

- 记录 token、延迟、provider metadata
- 不把具体模型调用写死到 operator 外部

## Task 3.3：实现 `synthesize_from_seed`

状态：

- `TODO`

归属任务包：

- `T5`

目标：

- 为后续数据合成提供可扩展入口。

验收标准：

- 输入种子样本
- 输出合成样本与生成元数据

## Task 3.4：实现 `StrategyRegistry`

状态：

- `IN_PROGRESS`

当前情况：

- `registry/strategy_registry.py` 已经实现了通用 registry，但还没有被 planner / controller 的实际选择流程消费

归属任务包：

- `T3`

目标：

- 把 planner/controller 的策略与 operator 解耦。

建议产出：

- `registry/strategy_registry.py`

验收标准：

- 未来可独立注册 rule-based 或 LLM-based 策略

## Task 3.5：增强成本与反馈建模

状态：

- `TODO`

归属任务包：

- `T5`

目标：

- 为 L3 优化提供更细粒度的成本和收益信号。

建议内容：

- token 成本
- API 成本
- CPU / GPU 时间
- 数据收益估计

验收标准：

- controller 可基于成本和效果做联合决策

## Task 3.6：为 Planner / Controller 预留 Agentic 升级口

状态：

- `IN_PROGRESS`

当前情况：

- `Planner`、`Controller`、`Runtime` 的边界协议已经存在，但还没有通过真实可替换策略完成端到端验证

归属任务包：

- `T3`

目标：

- 允许未来把规则版实现替换为更智能的版本，而不改动 runtime 和 storage。

验收标准：

- `Planner` / `Controller` 都通过稳定接口暴露
- runtime 不依赖具体规划策略

## Phase 3 完成定义

以下条件满足时，说明系统已经具备 agentic 升级基础：

- rewrite / synthesis 已不依赖单一 provider
- planner / controller 的策略可替换
- 成本与反馈接口足够支持更智能的优化器接入
