# Phase 0：抽象与边界冻结

## 阶段目标

这一阶段只做一件事：把 DataOS 的语言、骨架和边界定义清楚。先冻结抽象，再进入实现。

当前进度：

- `7 / 7` 已完成

补充：

- `2026-03-19` 已完成一次 `T0` 基线收口，冻结了内置 `operator_id` 风格、`simple_sft_eval` 角色和 `planner / runtime / eval / controller` handoff

## Task 0.1：建立 `dataos/` 包目录

状态：

- `DONE`

目标：

- 在仓库中新增独立 `dataos/` 包，与现有 `py_agent/` 解耦。

建议产出：

- `dataos/__init__.py`
- `dataos/core/`
- `dataos/registry/`
- `dataos/operators/`
- `dataos/planner/`
- `dataos/controller/`
- `dataos/runtime/`
- `dataos/storage/`
- `dataos/eval/`
- `dataos/experiments/`

验收标准：

- 包结构清晰
- 不影响现有 `py_agent/`

## Task 0.2：定义基础枚举与命名

状态：

- `DONE`

目标：

- 统一系统中的关键分类维度和命名。

建议产出：

- `core/enums.py`

至少包含：

- `TrainingStage`
- `IntelligenceLevel`
- `TaskType`
- `OperatorCategory`
- `StepStatus`

验收标准：

- 上层任务和下层算子共享同一套枚举定义
- 不出现重复或冲突命名

## Task 0.3：定义 `TaskSpec`

状态：

- `DONE`

目标：

- 建立任务声明式 schema，让上层可以表达数据任务。

建议产出：

- `core/task.py`

建议字段：

- `task_id`
- `name`
- `training_stage`
- `intelligence_level`
- `task_type`
- `input_schema`
- `output_schema`
- `constraints`
- `objective`
- `budget`
- `stop_condition`
- `eval_spec`
- `allowed_operators`
- `metadata`

验收标准：

- 能完整表达一个 SFT 数据任务
- 不依赖具体算法实现

## Task 0.4：定义 `DatasetVersion`

状态：

- `DONE`

目标：

- 建立数据版本对象和 lineage 结构。

建议产出：

- `core/dataset.py`

建议字段：

- `dataset_id`
- `version`
- `source`
- `schema`
- `statistics`
- `tags`
- `lineage`
- `storage_uri`
- `created_at`
- `created_by_run`

验收标准：

- 可追踪版本、来源和上游依赖
- 能承接后续存储层落盘

## Task 0.5：定义 `Operator` 抽象

状态：

- `DONE`

目标：

- 建立统一可插拔算子接口。

建议产出：

- `core/operator.py`

建议包含：

- `OperatorSpec`
- `OperatorContext`
- `OperatorResult`
- `Operator.execute()`

验收标准：

- 所有算子都能通过统一接口接入 registry 和 runtime

## Task 0.6：定义运行中间对象

状态：

- `DONE`

目标：

- 补齐 pipeline、eval、optimization 需要的公共对象。

建议产出：

- `core/pipeline.py`
- `core/eval.py`
- `core/optimization.py`

至少包含：

- `PipelineStep`
- `EvalResult`
- `OptimizationState`

验收标准：

- 运行、评测和优化状态都能结构化表达

## Task 0.7：冻结模块职责边界

状态：

- `DONE`

目标：

- 把 `core / registry / runtime / storage / planner / controller / eval / operators` 的边界写清楚。

建议产出：

- 更新设计文档
- 在包目录中按职责建立占位模块

验收标准：

- 后续实现时不会出现 planner/controller/runtime 职责混淆

## Phase 0 完成定义

以下条件全部满足后，Phase 0 才算完成：

- 核心 schema 已定义
- 目录结构已建立
- 模块边界已说明
- 后续开发可直接按对象和模块推进
