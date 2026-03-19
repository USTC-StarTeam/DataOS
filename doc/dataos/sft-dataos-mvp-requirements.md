# Agentic DataOS SFT MVP 需求文档

## 1. 背景与目标

Agentic DataOS 的目标不是做一组零散的数据处理脚本，而是构建一套面向训练数据工作的可扩展数据系统。系统要把数据工作从“人工经验驱动”升级为“任务定义 + 算子执行 + 评测反馈 + 闭环优化”。

当前阶段只做 SFT 数据场景的 MVP，优先验证以下能力可以跑通：

- 数据清洗
- 数据筛选 / 选择
- 数据混合
- 数据增强 / 合成
- 数据重写
- 数据配比
- 评测反馈驱动的迭代优化

MVP 的优先级是：

- 结构正确
- 接口清晰
- 可复现
- 可插拔
- 可追踪 lineage

## 2. 当前阶段边界

### In Scope

- 仅支持 `TrainingStage = sft`
- 支持任务类型：
  - `clean`
  - `select`
  - `mix`
  - `synthesize`
  - `rewrite`
  - `evaluate`
  - `iterate`
- 先提供规则版 planner 与有限轮 controller
- 先提供本地单机顺序执行 runtime
- 先支持离线代理评测，不要求训练大规模模型
- 先支持 `jsonl`，并为 `parquet` 预留接口

### Out of Scope

- 预训练 / 持续预训练 / RL / Eval 全阶段统一优化
- 复杂多智能体协商
- 重量级分布式调度系统
- 复杂 UI 或控制台
- 某篇论文方法的一次性硬编码复现

## 3. 四层架构映射

### L1：核心分类层

用于描述任务和实验的元信息，不承载复杂运行逻辑。

必须保留两个维度：

- `TrainingStage`: `pretrain | continual_pretrain | sft | rl | eval`
- `IntelligenceLevel`: `process | optimize_with_eval | closed_loop_agentic`

要求：

- L1 更像标签体系和实验上下文
- 不能实现成复杂推理器

### L2：任务映射层

负责把抽象系统映射为具体任务规格。

任务类型至少包括：

- `clean`
- `select`
- `mix`
- `synthesize`
- `rewrite`
- `evaluate`
- `iterate`

每个任务必须显式描述：

- 输入数据
- 输出数据
- 约束
- 优化目标
- 可调用算子

要求：

- L2 是声明式规格层
- 不能把具体论文方法直接硬编码为任务

### L3：优化与控制层

负责系统级决策与闭环优化。

必须负责：

- 根据 `TaskSpec` 生成执行计划
- 决定算子与执行顺序
- 调用评测器获取反馈
- 基于反馈做迭代
- 管理预算、轮次、停止条件

MVP 要求：

- 先做规则版 planner
- 先做有限轮 controller
- 为未来 agentic planner 预留策略接口

### L4：Runtime / 实现层

负责执行底座与工程能力。

必须负责：

- 算子注册与调用
- 工作流编排
- agent/tool 执行
- 中间状态保存
- 日志、指标、产物落盘
- 恢复与重试

要求：

- Runtime 必须可插拔
- 规划、执行、评测、存储要解耦
- 不能写成一个巨型 monolithic agent

## 4. MVP 最小闭环

MVP 需要完整跑通以下链路：

1. 用户定义一个 SFT 任务。
2. 系统读取候选数据集。
3. 系统执行若干数据算子。
4. 系统生成新的训练集版本。
5. 系统触发离线代理评测。
6. 系统根据评测结果继续优化。
7. 系统输出最佳数据版本、评测结果与完整 lineage。

## 5. 核心抽象要求

以下抽象必须优先定义清楚，再开始实现具体算子。

### 5.1 TaskSpec

建议字段：

- `task_id`
- `name`
- `training_stage`
- `task_type`
- `input_schema`
- `output_schema`
- `constraints`
- `objective`
- `budget`
- `stop_condition`
- `eval_spec`

### 5.2 Dataset

建议字段：

- `dataset_id`
- `version`
- `source`
- `schema`
- `statistics`
- `tags`
- `lineage`
- `storage_uri`

### 5.3 Operator

建议字段：

- `operator_id`
- `name`
- `category`
- `input_type`
- `output_type`
- `config_schema`
- `execute()`

至少支持类别：

- `cleaning`
- `selection`
- `scoring`
- `mixing`
- `synthesis`
- `rewriting`
- `evaluation`
- `utility`

### 5.4 PipelineStep

建议字段：

- `step_id`
- `operator_id`
- `inputs`
- `config`
- `outputs`
- `status`
- `metrics`

### 5.5 EvalResult

建议字段：

- `eval_id`
- `dataset_version`
- `task_id`
- `metrics`
- `artifacts`
- `summary`

### 5.6 OptimizationState

建议字段：

- `round_id`
- `candidate_configs`
- `tried_configs`
- `best_config`
- `best_result`
- `history`

## 6. 系统行为约束

### 6.1 声明式优先

- 任务、数据、算子、评测优先使用声明式配置表达
- 避免把主流程写死在层层 `if-else` 中

### 6.2 算子可插拔

- 任意论文方法都应尽量下沉为 operator 或 strategy
- 不把论文方法绑定为不可复用 pipeline

### 6.3 评测闭环

- 数据处理后不能直接结束
- 系统必须保留 `evaluate -> analyze -> refine` 的能力

### 6.4 强可观测性

每一步必须记录：

- 输入版本
- 输出版本
- 参数
- 耗时
- 关键指标
- 错误信息

### 6.5 强可复现性

- 相同任务定义、算子配置、随机种子，应可复现相同结果

### 6.6 成本意识

每一步需预留成本记录接口：

- token 成本
- API 成本
- CPU / GPU 成本
- 总运行时

### 6.7 失败恢复

- 长流程必须支持 checkpoint、resume、retry
- 运行状态不能只存在内存中

## 7. 推荐模块划分

- `core/`
  - 核心抽象与 schema
- `registry/`
  - operator、evaluator、strategy 注册中心
- `operators/`
  - 具体算子实现
- `planner/`
  - 根据任务规格生成 pipeline
- `controller/`
  - 管理优化轮次与结果比较
- `runtime/`
  - 执行 pipeline、保存状态、恢复失败
- `storage/`
  - 管理数据版本、指标、日志、产物与缓存
- `eval/`
  - 实现评测接口
- `experiments/`
  - 存放任务配置和实验入口

## 8. 第一版推荐能力

优先实现：

1. `TaskSpec` schema
2. `Dataset` versioning
3. `Operator` 抽象与 registry
4. 顺序执行 runtime
5. 规则版 planner
6. 有限轮 controller
7. 一组 SFT 常用 runtime 算子：
   - `dedup`
   - `filter_by_rule`
   - `filter_by_score`
   - `rewrite_with_llm`
   - `mix_datasets`
   - `sample_by_strategy`
8. 一组 SFT evaluator：
   - `simple_sft_eval`
9. 日志与实验记录机制

## 9. 接口约束

### 模型提供方

- LLM 调用层不能写死到单一提供商
- 后续可接 OpenAI、开源模型、本地 vLLM

### 数据格式

- 不能把数据格式写死
- 至少兼容 `jsonl / parquet`

### 执行环境

- 本地单机先跑通
- 接口上允许后续切换到分布式执行

### 任务领域

- 当前只聚焦 SFT
- 代码不能写成只服务某个 benchmark

## 10. 交付要求

本轮实现前，先交付并对齐以下内容：

1. 项目目录结构
2. 核心类与接口定义
3. 一个最小可运行样例设计
4. 每个模块的职责边界
5. 哪些部分是规则版，哪些部分未来升级为 agentic 版

## 11. 实现哲学

这个系统的本质是：

- 把数据任务对象化
- 把论文方法算子化
- 把实验过程闭环化
- 把运行系统工程化

最终需要满足：

- 上层可表达任务
- 中层可优化决策
- 下层可稳定执行
- 全程可记录、可比较、可迭代
