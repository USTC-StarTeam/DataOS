# Phase 1：单轮闭环跑通

## 阶段目标

这一阶段要把 SFT MVP 的最小闭环真正跑通，让系统从“只有抽象”进入“可以处理数据并给出评测结果”。

当前进度：

- `5 / 10` 已完成

补充说明：

- 当前代码里 `DatasetStore`、`RunStore`、`OperatorRegistry`、顺序 `Runtime Executor`、规则版 `Planner` 已经落地并有回归测试。
- 这一阶段的执行现在拆成两个任务包：
  - `T1 闭环算子包`：负责 `1.6` 到 `1.9`
  - `T4 示例与验收包`：负责 `1.10`

## 本阶段前置验证

在开始补齐本阶段剩余任务前，先完成 `T0` 的共享合同验证：

- 统一 `operator_id` 命名风格
- 冻结 `simple_sft_eval` 的角色定义
- 确认 `planner -> runtime -> eval` 之间的边界对象不再变化

当前状态：

- 已完成，冻结结论见 `doc/dataos/t0-baseline-contract.md`

## 推荐 worktree 划分

- `T1`：
  - 单开一个 worktree，集中实现基础 SFT 算子和闭环 evaluator
- `T4`：
  - 单开一个 worktree，等 `T1` 合入后再负责最小样例、运行脚本和集成闭环

## Task 1.1：实现 `DatasetStore`

状态：

- `DONE`

目标：

- 管理数据版本的保存与读取。

建议产出：

- `storage/dataset_store.py`

最低要求：

- 支持 `jsonl` 落盘和读取
- 为 `parquet` 预留接口
- 同步保存 metadata

验收标准：

- 能保存并读取版本化数据集

## Task 1.2：实现 `RunStore`

状态：

- `DONE`

目标：

- 保存 run 级计划、step 状态、metrics、eval 结果。

建议产出：

- `storage/run_store.py`

验收标准：

- 一次运行的关键状态可查询
- controller 和 runtime 都能复用

## Task 1.3：实现 `OperatorRegistry`

状态：

- `DONE`

目标：

- 支持算子注册、发现、查询。

建议产出：

- `registry/operator_registry.py`

验收标准：

- 能按 `operator_id` 获取算子实例
- 便于 planner 检查 allowed operators

## Task 1.4：实现顺序 `Runtime Executor`

状态：

- `DONE`

目标：

- 按 step 顺序执行 pipeline。

建议产出：

- `runtime/executor.py`

验收标准：

- 能执行多步 pipeline
- 每步都记录输入、输出、配置、耗时与 metrics

## Task 1.5：实现规则版 `Planner`

状态：

- `DONE`

目标：

- 根据 `TaskSpec` 生成最小规则版 pipeline。

建议产出：

- `planner/rule_based_planner.py`

第一版建议：

- `clean` 任务映射为 `filter_by_rule -> dedup`
- `select` 任务映射为 `filter_by_score -> sample_by_strategy`
- `iterate` 任务映射为 `clean/select -> evaluate`

验收标准：

- 至少支持 SFT MVP 样例生成可执行计划

## Task 1.6：实现 `filter_by_rule`

状态：

- `TODO`

归属任务包：

- `T1`

目标：

- 做基础质量过滤。

建议规则：

- 空字段
- 文本长度上下限
- 非法标签
- 黑名单来源

验收标准：

- 输出保留率
- 输出过滤原因统计

## Task 1.7：实现 `dedup`

状态：

- `TODO`

归属任务包：

- `T1`

目标：

- 去掉重复样本。

建议策略：

- 按 `id` 去重
- 按归一化 `prompt + response` 去重

验收标准：

- 输出去重前后样本数
- 结果可复现

## Task 1.8：实现 `filter_by_score`

状态：

- `TODO`

归属任务包：

- `T1`

目标：

- 基于样本分数做筛选。

最低要求：

- 支持阈值配置
- 输出分数分布统计

验收标准：

- 可根据阈值稳定筛选样本

## Task 1.9：实现 `simple_sft_eval`

状态：

- `TODO`

归属任务包：

- `T1`

目标：

- 提供离线代理评测，用于反馈闭环。

前置约束：

- 已在 `T0` 中冻结为 evaluator，并与 runtime pipeline 解耦

建议指标：

- 样本数
- 平均长度
- 来源分布
- score 均值 / 分位数
- 规则命中率

验收标准：

- 生成结构化 `EvalResult`
- controller 可以读取这些指标继续决策

## Task 1.10：打通最小单轮闭环

状态：

- `BLOCKED`

归属任务包：

- `T4`

目标：

- 把 `TaskSpec -> Planner -> Runtime -> Eval -> Storage` 全链路串起来。

建议样例链路：

1. 读取候选数据集
2. `filter_by_rule`
3. `dedup`
4. `filter_by_score`
5. runtime 输出候选数据版本
6. `simple_sft_eval`
7. 保存新版本数据和 run artifacts

验收标准：

- 能从候选数据集产出新的版本化训练集
- 能落盘 metrics、eval 和 lineage 基础信息

## Phase 1 完成定义

以下条件全部满足后，Phase 1 才算完成：

- 最小 pipeline 可以执行
- 至少三个基础算子可用
- 评测结果可结构化保存
- 新数据版本可追踪和可复现
