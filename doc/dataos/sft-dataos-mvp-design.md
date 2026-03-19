# Agentic DataOS SFT MVP 架构设计

## 1. 设计目标

本设计面向 SFT 数据闭环 MVP，目标是优先建立一个稳定、可复现、可插拔、可追踪的系统骨架，而不是一次性实现完整的 agent 世界观。

设计原则：

- 任务声明式
- 算子插件化
- 评测闭环化
- Runtime 工程化
- 存储与执行解耦
- 规则版先行，Agentic 预留升级口

## 2. 推荐目录结构

```text
dataos/
  core/
    enums.py
    schemas.py
    dataset.py
    task.py
    operator.py
    pipeline.py
    eval.py
    optimization.py
  registry/
    operator_registry.py
    evaluator_registry.py
    strategy_registry.py
  operators/
    cleaning/
      dedup.py
      filter_by_rule.py
    selection/
      filter_by_score.py
      sample_by_strategy.py
    mixing/
      mix_datasets.py
    synthesis/
      synthesize_from_seed.py
    rewriting/
      rewrite_with_llm.py
    utility/
      io.py
      statistics.py
  planner/
    base.py
    rule_based_planner.py
  controller/
    base.py
    iterative_controller.py
  runtime/
    executor.py
    state_store.py
    checkpoint.py
    retry.py
  storage/
    dataset_store.py
    artifact_store.py
    metrics_store.py
    lineage_store.py
    run_store.py
  eval/
    base.py
    offline_proxy_eval.py
    simple_sft_eval.py
    result_analyzer.py
  llm/
    base.py
    adapters/
      openai.py
      vllm.py
  experiments/
    sft/
      tasks/
      configs/
      run_sft_mvp.py
  examples/
    sft_candidate.jsonl
```

说明：

- `dataos/` 是推荐新增的独立 Python 包，与现有 `py_agent/` 解耦。
- `llm/` 只提供统一调用抽象，不把具体 provider 绑定到上层算子。
- `experiments/` 只负责组织任务配置与实验入口，不承担基础设施职责。

## 3. 分层职责映射

### L1：核心分类层

建议对象：

- `TrainingStage`
- `IntelligenceLevel`
- `TaskType`
- `OperatorCategory`

职责：

- 定义系统元信息和标签
- 支撑实验追踪、过滤、统计、对比

不做：

- 不直接参与 runtime 决策
- 不承载复杂规划逻辑

### L2：任务映射层

建议对象：

- `TaskSpec`
- `EvalSpec`
- `BudgetSpec`
- `StopCondition`

职责：

- 声明任务要解决什么问题
- 规定输入、输出、约束、优化目标、预算和评测方式
- 限制可用算子范围

不做：

- 不实现具体算子算法
- 不直接控制执行顺序

### L3：优化与控制层

建议对象：

- `Planner`
- `Controller`
- `OptimizationState`
- `ResultAnalyzer`

职责：

- 基于任务规格生成候选 pipeline
- 在每一轮中比较候选结果
- 基于评测反馈调整配置
- 维护预算、轮次、停止条件

MVP 规则版实现：

- `RuleBasedPlanner`
  - 按任务类型映射到固定 step 模板
- `IterativeController`
  - 在少量候选配置中搜索最佳版本

未来 Agentic 升级方向：

- LLM-assisted planner
- 基于历史实验的策略学习
- 多目标成本约束优化
- 自主提议新数据变换策略

### L4：Runtime / 实现层

建议对象：

- `PipelineExecutor`
- `StateStore`
- `CheckpointManager`
- `RetryPolicy`
- `DatasetStore`
- `RunStore`

职责：

- 执行 step
- 记录状态
- 存储数据版本和产物
- 处理失败恢复和重试

不做：

- 不承载高层任务规划
- 不把评测分析逻辑硬塞进 executor

## 4. 核心类与接口定义

### 4.1 TaskSpec

```python
@dataclass
class TaskSpec:
    task_id: str
    name: str
    training_stage: TrainingStage
    intelligence_level: IntelligenceLevel
    task_type: TaskType
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    constraints: dict[str, Any]
    objective: dict[str, Any]
    budget: dict[str, Any]
    stop_condition: dict[str, Any]
    eval_spec: EvalSpec
    allowed_operators: list[str]
    metadata: dict[str, Any]
```

说明：

- `allowed_operators` 用于限制 planner 的搜索空间
- `budget` 与 `stop_condition` 是 L3 的控制输入

### 4.2 Dataset

```python
@dataclass
class DatasetVersion:
    dataset_id: str
    version: str
    source: dict[str, Any]
    schema: dict[str, Any]
    statistics: dict[str, Any]
    tags: list[str]
    lineage: list[dict[str, Any]]
    storage_uri: str
    created_at: str
    created_by_run: str
```

说明：

- `lineage` 记录上游版本、算子、配置和产物位置
- `statistics` 用于评测前后的快速比较

### 4.3 Operator

```python
class Operator(Protocol):
    spec: OperatorSpec

    def execute(self, context: OperatorContext) -> OperatorResult:
        ...
```

```python
@dataclass
class OperatorSpec:
    operator_id: str
    name: str
    category: OperatorCategory
    input_type: str
    output_type: str
    config_schema: dict[str, Any]
    deterministic: bool
```

说明：

- `Operator` 只关心单步转换
- 算子必须显式声明输入/输出类型与配置 schema

### 4.4 PipelineStep

```python
@dataclass
class PipelineStep:
    step_id: str
    operator_id: str
    inputs: list[str]
    config: dict[str, Any]
    outputs: list[str]
    status: str
    metrics: dict[str, Any]
```

### 4.5 EvalResult

```python
@dataclass
class EvalResult:
    eval_id: str
    dataset_version: str
    task_id: str
    metrics: dict[str, float]
    artifacts: dict[str, str]
    summary: dict[str, Any]
```

### 4.6 OptimizationState

```python
@dataclass
class OptimizationState:
    round_id: str
    candidate_configs: list[dict[str, Any]]
    tried_configs: list[dict[str, Any]]
    best_config: dict[str, Any] | None
    best_result: EvalResult | None
    history: list[dict[str, Any]]
```

## 5. 模块职责边界

这一节是 Phase 0 的冻结基线。后续实现可以替换具体算法，但不能突破这里定义的职责边界。

### 5.1 模块输入输出矩阵

| 模块 | 核心输入 | 核心输出 | 负责什么 | 明确不负责 |
| --- | --- | --- | --- | --- |
| `core` | 无基础设施依赖，只接收通用 Python 数据 | `TaskSpec`、`DatasetVersion`、`PipelineStep`、`EvalResult`、`OptimizationState` 等稳定对象 | 定义全系统共享 schema、枚举和协议承载对象 | 不接触文件系统、模型 provider、网络、具体算子实现 |
| `registry` | operator / evaluator / strategy 实例或 spec | 可查询的注册表视图 | 管理注册、发现、过滤、按 ID 解析 | 不执行 step、不持久化运行状态、不做策略决策 |
| `operators` | `OperatorContext` | `OperatorResult` | 完成单步数据变换、打分或辅助处理 | 不做多轮优化、不管理全局预算、不决定下一步 pipeline，也不产出 `EvalResult` |
| `planner` | `TaskSpec`、候选 `DatasetVersion`、`OperatorRegistry`、历史 `OptimizationState` | `PlanningResult`，本质上是待执行的 step 序列和候选配置 | 生成或调整 pipeline 计划，缩小搜索空间 | 不执行 step、不写入 dataset、不判断是否终止整轮优化 |
| `runtime` | `RuntimeExecutionRequest`、`PlanningResult.pipeline`、`OperatorRegistry`、`Storage` 接口 | `RuntimeExecutionResult`、step 状态、checkpoint、副产物 | 可靠执行给定计划，处理 step 生命周期、失败恢复和重试 | 不新增/删改计划步骤、不选择候选策略、不决定是否继续下一轮 |
| `storage` | `DatasetVersion`、运行记录、指标、artifact、lineage | 可持久化和可回读的数据、索引和引用 URI | 保存 dataset、run、metrics、artifacts、lineage | 不定义业务规则、不执行算子、不做 planner/controller 决策 |
| `eval` | `TaskSpec`、候选 `DatasetVersion`、必要 artifact | `EvalResult`、`AnalysisSummary` | 评估候选数据版本并生成可比较反馈 | 不直接改写数据、不调度运行闭环 |
| `controller` | `TaskSpec`、`OptimizationState`、`PlanningResult`、`RuntimeExecutionResult`、`EvalResult` | `ControllerDecision`、更新后的 `OptimizationState` | 编排“计划 -> 执行 -> 评测 -> 是否继续”的多轮闭环 | 不直接执行 operator、不直接写 dataset payload、不替 runtime 做恢复细节 |

### 5.2 三条硬边界

- `planner` 只决定“尝试什么”，不能决定“如何执行得更稳”，也不能直接调用 `Operator.execute()`。
- `runtime` 只决定“既定计划如何执行”，不能决定“下一轮还要不要继续”或“是否换一组候选策略”。
- `controller` 只决定“闭环如何推进”，不能跳过 `runtime` 直接做数据变换，也不能把 dataset 落盘逻辑塞进自己。
- `evaluator` 不能伪装成 runtime operator；`simple_sft_eval` 这类评测能力必须经由 `eval` 层产生 `EvalResult`。

### 5.3 允许依赖方向

- `core` 是最底层，其他模块都可以依赖它。
- `operators` 依赖 `core`，必要时依赖 `llm`，但不依赖 `planner`、`controller`、`runtime`。
- `registry` 依赖 `core` 以及被注册对象的最小接口，不反向依赖 `runtime`。
- `planner` 依赖 `core` 和 `registry`，可以读取 `OptimizationState`，不能依赖具体 `storage` 落盘细节。
- `runtime` 依赖 `core`、`registry`、`storage`，但不依赖具体 planner/controller 策略实现。
- `eval` 依赖 `core`，可选依赖 `storage` 和 `llm`，但不承担执行职责。
- `controller` 依赖 `planner`、`runtime`、`eval`、`storage` 的接口来编排闭环，但不下沉为具体算子实现层。

### 5.4 模块补充说明

#### core/

职责：

- 保存稳定的领域抽象
- 为 planner/runtime/controller/storage 提供共同语言

边界：

- 不直接依赖具体 operator 实现
- 不直接操作文件系统或模型提供商

#### registry/

职责：

- 管理 operator、evaluator、strategy 的注册、发现与按 ID 解析
- 为 planner 提供可搜索的能力边界

边界：

- 不负责执行
- 不保存运行时状态

#### operators/

职责：

- 封装具体数据变换、打分和辅助处理算子
- 只暴露单步、可组合的执行单元

边界：

- 单个算子只做一件事
- 不处理多轮全局优化

#### planner/

职责：

- 根据 `TaskSpec`、可用 operator 和历史状态生成候选 plan
- 输出的是“计划草案”，不是“执行结果”

边界：

- 只生成计划，不执行计划
- 不更新 dataset version 或 run state

#### controller/

职责：

- 管理多轮优化、结果比较、预算与停止判断
- 负责在 planner/runtime/eval 之间传递上下文

边界：

- 不直接读写底层数据文件
- 不把 step 执行细节塞进自身

#### runtime/

职责：

- 执行 step
- 记录 step 状态
- 管理 checkpoint / resume / retry

边界：

- 不决定高层任务目标
- 不负责实验策略搜索

#### storage/

职责：

- 统一管理 dataset version、metrics、artifacts、logs、lineage 和 run record
- 为 runtime/controller/eval 提供持久化接口

边界：

- 不写业务规则
- 不承担 planner/controller 决策

#### eval/

职责：

- 提供评测器与评测结果分析器
- 输出 controller 可消费的结构化反馈

边界：

- 不直接负责数据转换
- 不调度多轮闭环

#### experiments/

职责：

- 组织任务配置、实验脚本与样例数据
- 组合已有模块形成场景入口

边界：

- 不应承载基础抽象和长期复用逻辑
- 不重新定义 planner/runtime/controller 的边界

## 6. MVP 运行闭环

建议的最小执行流程如下：

1. `Controller` 读取 `TaskSpec`、历史 `OptimizationState` 和候选 `DatasetVersion`
2. `Controller` 调用 `Planner` 生成一个或多个 `PlanningResult`
3. `Controller` 选择当前轮要执行的候选 plan，并为本轮分配 `run_id`
4. `Runtime` 按 `PlanningResult.pipeline` 顺序执行 step，并通过 `Storage` 写入 run state / checkpoint / dataset artifact
5. `Runtime` 返回 `RuntimeExecutionResult`，其中包含 step 状态、输出数据版本引用和执行副产物
6. `Eval` 基于候选输出数据版本生成 `EvalResult` 和分析摘要
7. `Controller` 合并 `EvalResult`、更新 `OptimizationState`，判断继续、重规划还是终止
8. `Storage` 持久化最佳版本、指标、产物和 lineage；若终止则由 `Controller` 输出最终决策

## 7. MVP 最小 runtime operator 与 evaluator 集合

### cleaning

- `dedup`
  - 基于样本键或归一化文本去重
- `filter_by_rule`
  - 按长度、空字段、非法标签等规则过滤

### selection

- `filter_by_score`
  - 根据已有 score 字段或外部打分结果筛选
- `sample_by_strategy`
  - 按桶、标签、来源或难度采样

### mixing

- `mix_datasets`
  - 按配比合并多个数据版本

### synthesis

- `synthesize_from_seed`
  - 基于种子样本扩充数据

### rewriting

- `rewrite_with_llm`
  - 通过统一 LLM 接口进行重写

### eval

- `simple_sft_eval`
  - 作为 evaluator 使用离线代理指标进行 quick feedback

## 8. 可观测性与可复现设计

每个 step 至少记录：

- `run_id`
- `step_id`
- 输入数据版本
- 输出数据版本
- operator 配置
- 随机种子
- 开始和结束时间
- 耗时
- 关键统计指标
- token / API / CPU / GPU 成本
- 错误信息

建议产物布局：

```text
artifacts/
  runs/<run_id>/
    plan.json
    state.json
    steps/<step_id>.json
    metrics.json
    eval.json
  datasets/<dataset_id>/<version>/
    data.jsonl
    metadata.json
    lineage.json
```

## 9. 失败恢复设计

MVP 需要至少支持以下恢复能力：

- step 级 checkpoint
- run 级 resume
- operator 级 retry
- 幂等写入保护

恢复策略建议：

- step 成功后立即持久化输出元数据
- controller 每轮结束后持久化 `OptimizationState`
- 同一配置重复执行时优先命中缓存或历史产物

## 10. 规则版与未来 Agentic 升级边界

### 当前按规则实现的部分

- `RuleBasedPlanner`
- `IterativeController` 的候选配置搜索
- `simple_sft_eval` evaluator 占位
- 基于本地文件系统的 `Storage`
- 顺序执行 `Runtime`

### 未来升级为 Agentic 的部分

- planner 从固定模板升级为 LLM-assisted 计划生成
- controller 从人工阈值升级为基于反馈的策略优化
- evaluator 从离线代理分数升级为真实训练反馈和在线评测
- runtime 从顺序执行升级为并行 / 分布式执行
- synthesis / rewriting 从单算子调用升级为工具使用型 agent

## 11. 最小可运行样例设计

MVP 样例建议如下：

输入：

- 一个候选 `jsonl` 数据集，字段至少包含：
  - `id`
  - `prompt`
  - `response`
  - `source`
  - `score`

执行步骤：

1. `filter_by_rule`
2. `dedup`
3. `filter_by_score`
4. `mix_datasets` 或 `sample_by_strategy`
5. runtime 输出候选数据版本
6. `simple_sft_eval`

输出：

- 新版本训练集
- 评测结果
- run 级 metrics
- lineage 记录

这个样例足以验证：

- 声明式任务定义
- 算子注册与执行
- 版本化数据产出
- 评测闭环
- 结果落盘与可追踪性
