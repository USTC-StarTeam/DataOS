## 系统实现

### 1. 总体实现原则

DataOS 当前实现遵循一个非常明确的分层原则：LLM 负责规划与复盘，确定性运行时负责执行，所有中间结果以 typed artifact 的方式持久化到本地磁盘。  
这意味着，系统虽然在设计上面向 Agentic DataOS，但在工程实现上仍然保持一个窄而稳定的执行边界：

- 数据主通道固定为 Hugging Face `Dataset` / `DatasetDict` 的 `save_to_disk` 目录。
- LLM 不直接执行数据处理，也不直接写 shell workflow；它只输出结构化 `plan` 和 `review` JSON。
- 真正执行数据操作的是本地注册过的 operator。
- 每轮实验都在独立 sandbox 中完成，并落盘 `plan.json`、`report.json`、`review.json` 和 artifact 元数据。

因此，当前版本更准确地说是一个面向 SFT 数据优化的 `L3-ready but runtime-constrained` 系统：控制面已经具备 agentic 闭环雏形，但执行面仍然是固定算子集、固定运行时、单轮实验。

### 2. 分层架构

从当前仓库实现看，DataOS 可以分成六个核心层次。

#### 2.1 任务规格层：Task Compiler 的最小实现

任务规格由 `dataos/core/task.py` 中的 `TaskTemplate` 承载，其作用是把上层任务定义编译成运行时可消费的 contract。当前 schema 包含：

- 任务身份：`task_id`、`name`
- 目标描述：`goal`
- 输入输出约束：`input_description`、`output_description`
- 约束与预算：`constraints`、`budget`
- 搜索边界：`max_rounds`、`pipeline_hint`
- 训练与评测配置：`train_config`、`eval_config`

这一步对应前文中的 Task Compiler 角色。它的作用不是“思考如何优化”，而是把任务目标、预算、约束和预期输出显式化，形成后续 Planner、Executor、Reviewer 共享的统一输入。

在当前 V0.1 中，`TaskTemplate` 还没有覆盖完整的机器可执行约束语言，例如风险预算、领域标签、禁止操作集合、评测 slice 定义等；但它已经具备最小可扩展骨架，可继续向更完整的 task contract 演进。

#### 2.2 控制面：Planning Agent

控制面由 `dataos/core/agent.py` 中的 `LLMPlanningAgent` 实现。它承担两个职责：

1. `plan()`：基于任务规格、初始 artifact 摘要、历史实验摘要和当前 operator registry，生成结构化 pipeline。
2. `review()`：基于实验记录和 artifact 链路，输出结构化复盘结果。

这里的关键设计点不是“让 LLM 足够聪明”，而是把 LLM 严格限制在一个可验证的 JSON 边界内：

- 规划 prompt 只允许使用 registry 中已注册的 operator。
- step 数量上限固定为 `PLAN_MAX_STEPS = 6`。
- 规划阶段只允许选择数据算子；训练和评测由固定运行时阶段在 plan 之后自动执行。
- 禁止发明新算子、禁止让 LLM 修改代码、禁止输出非 JSON 文本。

如果 LLM 输出非法，系统会退化到一个确定性的 fallback plan：

`clean_data -> synthesize_data`

这实际上体现了当前 DataOS 的一个重要工程哲学：  
LLM 可以提供策略建议，但不能突破运行时安全边界；当建议不可靠时，系统必须能回退到稳定的默认 pipeline。

#### 2.3 执行面：固定运行时 + Operator Registry + 训练评测阶段

执行面由三部分构成：

- `dataos/runtime/runner.py` 中的 `run_task()`
- `dataos/core/registry.py` 中的 `OperatorRegistry`
- `dataos/core/operator.py` 中定义的 `Operator` 接口
- `dataos/train/*` 与 `dataos/eval/*` 中的运行时训练/评测函数

`run_task()` 是系统主入口。它负责串起整个控制流：

1. 创建本地 artifact store。
2. 加载 `TaskTemplate`。
3. 创建实验 sandbox。
4. 将输入数据集 materialize 到 sandbox 的 `inputs/` 目录。
5. 构建默认 operator registry。
6. 创建 `LLMPlanningAgent`。
7. 创建 `Experiment`。
8. 调用 agent 生成 plan。
9. 逐步执行 plan 中的 operator。
10. 自动执行固定 runtime stage：训练再评测。
11. 生成 experiment artifact、review 和归档结果。

当前默认 registry 在 `build_default_registry()` 中只固定注册数据算子：

1. `clean_data`
2. `synthesize_data`

训练与评测则通过固定 runtime stage 追加执行：

1. `train_llamafactory`
2. `eval_model`

这说明现阶段 DataOS 的“搜索空间”仍是一个有限、显式、白名单化的数据 operator 集合。  
换句话说，当前系统支持的是：

- 在固定算子库内做有限 pipeline 选择
- 通过 step params 做有限参数配置

当前系统尚不支持：

- 动态编写新 operator 并自动注册
- 在线扩展 registry
- 多轮实验中的 program rewrite
- 跨实验的自动策略搜索与淘汰

因此，从 L1/L2/L3 的视角看，当前实现最接近：

- 在任务目标上具备 L3 的控制面接口
- 在执行机制上仍是受约束的 L1/L2.5 运行时

#### 2.4 数据面：Typed Artifact Model

数据面核心在 `dataos/core/artifact.py`。当前系统把所有中间结果和最终结果统一抽象为 artifact，并强制带类型、路径、父子 lineage 和元数据。

当前支持的 artifact type 固定为：

- `dataset`
- `model`
- `eval`
- `result`
- `experiment`

其中最关键的是 `DatasetArtifact`。它明确限定：

- 只能包装 Hugging Face `Dataset` 或 `DatasetDict`
- 底层存储必须是 `save_to_disk` 输出目录
- 元数据要从磁盘刷新并记录 `splits`、`num_rows`、`features`、`columns`、`fingerprint`

这套设计对应前文统一理论中的经验测度对象 `\mu_D`，但在工程上并不直接操作抽象测度，而是通过 artifact 元数据去近似表示“当前数据分布的一个可执行快照”。  
其价值在于：

- 让每一步 operator 都以明确输入输出类型运行
- 让 lineage 可追踪，便于回溯数据是如何从原始样本演化而来
- 让 planner/reviewer 可以读取 compact summary，而不需要直接处理原始数据文件

换言之，artifact model 是 DataOS 把“数据优化问题”转化为“可执行程序图”的关键桥梁。

#### 2.5 实验与状态层：Experiment

`dataos/core/experiment.py` 中的 `Experiment` 承担的是实验状态机和实验报告持久化职责。它维护：

- artifact 列表
- step 执行记录
- 状态字段：`created/running/completed/failed`
- `score`
- `error`
- 时间戳

每执行一个 operator，系统都会记录：

- `operator_name`
- `input_artifact_ids`
- `output_artifact_ids`
- `params`
- `status`
- `error`

实验结束后会写出 `report.json`，并生成 `ExperimentArtifact`。  
这让 DataOS 的每一轮运行都成为一个可复现、可分析、可归档的实验单元，而不是一次黑盒流水线执行。

从 agentic loop 的角度看，`Experiment` 相当于当前系统中 `H_t` 和 `A_t` 的最小落地形式：

- `H_t` 对应 step history、report、review
- `A_t` 对应 sandbox 中的输入、输出、日志、artifact 目录

当前版本还没有显式 memory subsystem，但 experiment 已经为未来的跨轮经验积累提供了天然载体。

#### 2.6 Sandbox 与本地持久化层

`dataos/core/sandbox.py` 定义了每轮实验的隔离执行环境。`Sandbox.prepare()` 会创建固定目录结构：

- `inputs/`
- `outputs/`
- `artifacts/`
- `logs/`
- `tmp/`
- `code/`

这套布局非常重要，因为它把“运行时约束”显式编码到了文件系统中：

- 输入数据放在 `inputs/`
- operator 输出 artifact 放在 `artifacts/`
- 日志写入 `logs/`
- 计划、报告、复盘写在 sandbox 根目录

实验完成后，`dataos/core/store.py` 中的 `LocalArtifactStore` 会把需要长期保留的 artifact 归档到 workspace 级目录，例如：

- `artifacts/datasets/`
- `artifacts/models/`
- `artifacts/evals/`
- `artifacts/results/`
- `artifacts/experiments/`

因此，DataOS 当前是一个典型的 local-first system：

- 执行在本地 sandbox 完成
- 状态在本地磁盘持久化
- artifact 通过目录复制归档
- 不依赖中心化数据服务才能完成单轮实验闭环

### 3. 当前内置算子如何映射到数据任务分类

虽然当前内置算子数量很少，但已经能映射到前文抽象出的三类核心数据任务。

#### 3.1 治理类算子：`clean_data`

`clean_data` 对应治理/清洗/去噪类任务。它当前执行的是最小治理集合：

- 字段名规范化
- 字符串去空白
- 删除空样本

它满足“尽量不改变任务语义，只做修复和规范化”的设计目标，因此对应前文中的治理算子 `G = (r, \phi)`。

#### 3.2 构造类算子：`synthesize_data`

`synthesize_data` 当前实现的是一个非常弱的 augmentation stub：对训练样本做比例复制，得到扩增后的数据集。  
它还不是一个真正基于 LLM 或规则系统的高质量 synthesis operator，但在抽象层面已经对应：

- 有锚点的数据增强
- 对原始分布进行 support-preserving 的局部扩展

因此它更接近构造类任务中的 augmentation，而不是无锚点 synthesis。

#### 3.3 训练与评测阶段：`train_llamafactory`、`eval_model`

这两个阶段不是数据操作本身，而是 DataOS 闭环里的反馈接口：

- `train_llamafactory` 把当前最新数据集转化为模型 artifact，并可在真实模式下调用 LLaMA-Factory 执行 SFT
- `eval_model` 产生 eval artifact 和最终 score

如果没有训练/评测阶段，系统就无法把“某种数据处理策略”映射回目标函数 `J`。  
因此在 DataOS 中，这两个阶段承担的是把数据优化闭环闭上的职责。

### 4. 控制流：从任务定义到闭环输出

当前仓库中的标准执行链条可以写成：

`TaskTemplate -> Plan -> Sandbox Execution -> Artifact Chain -> Report -> Review -> Archive`

更细粒度地说，一轮标准运行包含以下步骤：

1. 用户提供 task template 和初始数据集目录。
2. `run_task()` 把初始数据复制到 sandbox `inputs/` 下，生成首个 `DatasetArtifact`。
3. `LLMPlanningAgent.plan()` 读取任务、历史和可用数据算子描述，输出 plan JSON。
4. 运行时逐步执行 data operator，并把每个输出保存为新的 artifact。
5. 固定 runtime stage 自动执行训练和评测。
6. `Experiment` 持续记录 operator/runtime step history、状态和错误信息。
7. 运行结束后写出 `report.json`，并生成 `ExperimentArtifact`。
8. `LLMPlanningAgent.review()` 读取实验记录，输出 `review.json`。
9. `LocalArtifactStore` 将实验结果归档到 workspace。

这条链路已经具备前文标准闭环中的以下子集：

- `Spec`
- `Plan`
- `Run`
- `Evaluate`
- `Analyze`

但还缺少：

- 真正独立的 `Build`
- 自动化的 `Retrieve/Invent`
- 多轮 `Refine`
- 基于历史结果的持续 `Memory Update`

因此，当前系统可以被看作完整 agentic research loop 的第一阶段实现，即：  
先把“结构化任务定义 + 可验证计划 + 本地确定性执行 + 结构化复盘 + 全链路持久化”这条主干搭起来，再逐步增强搜索和自主演化能力。

### 5. 与完整 L3 Agentic DataOS 的差距

如果把前文 L3 定义为“系统能够围绕任务目标自主发现问题、提出策略、执行实验、读取反馈并持续迭代”，那么当前仓库还存在几个明确缺口。

#### 5.1 单轮而非多轮

`TaskTemplate.max_rounds` 已经存在，但当前 `run_task()` 只执行单次 experiment，尚未形成多轮 proposal-execute-review-refine 循环。

#### 5.2 搜索空间固定而非可生长

当前搜索对象仅限于：

- 预注册 operator 的选择
- operator params 的一次性设定

尚未支持：

- 动态生成 operator
- 从论文或 recipe 中抽象新 operator
- 将多步 workflow 封装为复合算子

#### 5.3 无外部知识检索

当前系统没有 `K_t` 对应的显式知识库，也没有 researcher 模块，因而还不能把“论文到算子”的过程纳入自动闭环。

#### 5.4 Builder 与 Experimenter 仍部分耦合

当前没有单独的 pipeline materialization 层。  
也就是说，plan 直接由 runtime 消费，而不是先编译成显式 executable spec、再交给 executor。这在 V0.1 中足够简单，但未来若要支持更复杂 pipeline、并行 stage 或可回滚 program graph，就需要把 Builder 层单独拆出。

### 6. 面向后续演进的实现路径

基于当前仓库结构，一个自然的演进方向不是推翻重来，而是在现有模块上逐层外扩。

#### 6.1 从固定计划执行到多轮闭环

可以在 `run_task()` 之上引入 round controller，把：

- `plan`
- `run_experiment`
- `review`
- `history_summaries` 更新

组织成显式多轮循环，并以预算、收益增量和风险阈值作为停止条件。

#### 6.2 从固定 registry 到可扩展 operator zoo

可以保留 `Operator` 抽象不变，但引入：

- operator capability schema
- operator versioning
- composite operator
- registry snapshot

从而支持算子搜索、回滚和跨实验复用。

#### 6.3 从实验记录到经验记忆

当前的 `report.json`、`review.json`、artifact lineage 和 step history，已经足以沉淀成：

- 失败模式库
- 参数经验库
- 数据任务模板库
- 论文方法到 operator motif 的映射库

这一步做完后，`history_summaries` 就不再只是被动输入，而会成为真正的长期 memory。

#### 6.4 从 stub 反馈到真实训练反馈

当前 `train_llamafactory` 已支持真实的 LLaMA-Factory SFT 训练路径，同时保留 `stub` 模式用于快速本地测试；`eval_model` 则已经重构成通用评测阶段，框架层只负责通用 artifact / generation / 落盘逻辑，GSM8K exact-match 作为第一个 backend 独立承载。它们都已经从 LLM 规划的 operator 层下沉为固定运行时阶段，并由独立函数模块承载。  
这意味着 DataOS 已经能够把“数据处理策略 -> 训练执行 -> 模型 artifact -> 标准测试集评估”闭回到本地实验记录中，后续可继续扩展到更多任务和指标。

### 7. 小结

综合来看，当前仓库中的 DataOS 已经实现了 Agentic DataOS 最关键的基础主干：

- 用 `TaskTemplate` 明确任务 contract
- 用 `LLMPlanningAgent` 生成和复盘结构化决策
- 用 `OperatorRegistry + Operator` 执行受控数据流程
- 用 `Artifact` 统一承载数据、模型、评测和实验状态
- 用 `Sandbox + LocalArtifactStore` 保证本地隔离执行与可归档持久化

所以，当前版本不是一个“已经完成的自主数据研究系统”，而是一个面向 SFT 场景、强调 local-first 和 fixed-runtime 的 Agentic DataOS 基座。  
它已经把 L3 所需的控制接口、执行边界和状态持久化机制搭好；后续需要补强的是多轮搜索、知识检索、算子发明和真实反馈执行能力。
