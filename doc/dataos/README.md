# Agentic DataOS SFT MVP 文档索引

本目录用于沉淀面向大模型训练数据工作的 Agentic DataOS 设计，当前阶段只覆盖 SFT 数据闭环 MVP。

文档清单：

- `t0-baseline-contract.md`
  - 记录 `T0` 收口后冻结的共享命名、模块边界和后续 worktree 写入约定。
- `sft-dataos-mvp-requirements.md`
  - 明确系统目标、MVP 范围、架构映射、接口约束与交付要求。
- `sft-dataos-mvp-design.md`
  - 定义目录结构、核心抽象、模块职责、运行闭环、可观测性与未来升级路径。
- `sft-dataos-mvp-task-breakdown.md`
  - 任务拆解总览，负责把整体路线图串起来。
- `tasks/`
  - 分阶段任务清单，按 Phase 进一步拆开实现项、依赖与验收标准。

建议阅读顺序：

1. 先看 `sft-dataos-mvp-requirements.md`，统一目标和边界。
2. 再看 `sft-dataos-mvp-design.md`，理解系统分层和模块职责。
3. 再看 `t0-baseline-contract.md`，对齐当前已经冻结的共享合同。
4. 再看 `sft-dataos-mvp-task-breakdown.md`，掌握整体推进顺序。
5. 最后进入 `tasks/` 目录，按阶段推进具体任务。
