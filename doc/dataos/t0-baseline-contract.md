# T0 基线合同冻结

这份文档记录 2026-03-19 在主工作区完成的 `T0` 收口结论，作为后续 `worktree` 的共享基线。

## 冻结结论

### 1. 内置 runtime `operator_id` 采用裸 `snake_case`

当前 SFT MVP 内置 runtime operator 统一使用：

- `filter_by_rule`
- `dedup`
- `filter_by_score`
- `sample_by_strategy`

说明：

- `cleaning.filter_by_rule`、`selection.filter_by_score` 这类带命名空间的写法不作为当前内置合同
- 自定义或外部 operator 仍然可以使用带点命名；但 DataOS SFT MVP 自带 operator 先冻结成裸 `snake_case`

### 2. `simple_sft_eval` 冻结为 evaluator

结论：

- `simple_sft_eval` 是 `evaluator_id`
- 它属于 `dataos.eval`
- 它不进入 `OperatorRegistry`
- 它不作为 `PlanningResult.pipeline` 中的 runtime step

约束：

- 如果以后需要“runtime 调 evaluator”的 adapter，必须使用新的 `operator_id`
- 不能再让 `simple_sft_eval` 同时承担 operator_id 和 evaluator_id 两套角色

### 3. `planner / runtime / eval / controller` 的边界按下列方式冻结

- `planner`
  - 只输出 runtime-only 的 `PipelineStep` 序列
  - 可以在 `PlanningResult.rationale` 中声明评测 handoff 信息
- `runtime`
  - 只执行 operator step
  - 只产出 `RuntimeExecutionResult`、step record、state、artifact 引用
- `eval`
  - 接收 `TaskSpec.eval_spec`、候选 `DatasetVersion` 和必要 artifact
  - 产出 `EvalResult` 和 `AnalysisSummary`
- `controller`
  - 负责串联 `planner -> runtime -> eval -> storage`
  - 负责是否继续、是否重规划、是否终止

## 当前落地位置

- `dataos/planner/rule_based_planner.py`
  - `pipeline` 不再注入 `simple_sft_eval`
  - `rationale["evaluation"]` 显式保留 post-runtime 评测 handoff
- `dataos/eval/simple_sft_eval.py`
  - 增加 `SimpleSFTEvaluator` 占位实现，明确角色归属
- `tests/`
  - 边界测试、planner 测试、run store 测试同步改为 runtime/eval 分离模型

## 后续 worktree 建议写入范围

- `T1`
  - `dataos/operators/**`
  - `dataos/eval/simple_sft_eval.py`
  - 相关 operator / planner / eval 测试
- `T2`
  - `dataos/runtime/**`
  - `dataos/storage/**`
- `T3`
  - `dataos/controller/**`
  - `dataos/eval/result_analyzer.py`
  - `dataos/registry/strategy_registry.py`
- `T4`
  - `dataos/examples/**`
  - `dataos/experiments/**`
  - 集成测试和样例文档
- `T5`
  - `dataos/llm/**`
  - `dataos/operators/rewriting/**`
  - `dataos/operators/synthesis/**`

这份合同如果需要调整，应当视为新的共享基线变更，而不是在后续功能 worktree 中顺手修改。
