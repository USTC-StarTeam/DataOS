# DataOS-Runtime 契约字段规范

本文件逐字段规定 `environments/<env>/env.yaml` 与 `tasks/<task>/task.yaml` **能写什么**。设计与时序见
`DataOS/docs/runtime-contract.md`。任何 agent 通过 `compose/compose.py` 的 `resolve()+render()` 消费本契约。

约定:类型用 `str / int / float / bool / list / map / path`;`path` 是字符串路径;命令模板是含 `{占位符}` 的
`str`(占位符见 §3)。**秘钥本身绝不入库**,只写 key 文件路径。

---

## 1. Environment 契约 —— `env.yaml`

| 字段 | 必填 | 类型 | 说明 / 允许值 |
|---|---|---|---|
| `name` | 是 | str | 环境标识,与目录名一致(如 `h100_slurm`) |
| `kind` | 是 | enum | `slurm` \| `k8s` \| `local`(调度类型;决定 operations 的语义) |
| `access` | 是 | map | 怎么到达该环境 |
| `access.host` | 是* | str | 主机(slurm/远程必填;local 可省) |
| `access.user` | 是* | str | 登录用户 |
| `access.port` | 否 | int | 默认 22 |
| `access.key_ref` | 否 | path | 控制面上的私钥路径(不入库);无则用默认 ssh 配置 |
| `tools` | 是 | map | 机器预装的**通用运行时**(uv venv / conda / bare venv 皆可);**只列通用框架/基础设施,不列任何单个 agent 的私有方法库**(如 dataos_operator) |
| `tools.<名>` | — | map | 每个运行时:`{path: <venv 或 conda prefix,须有 bin/python>, provides: [通用框架名/能力...]}`;一律按 `{path}/bin/<tool>` 调用,与环境管理器无关 |
| `paths` | 是 | map | 本部署的文件系统事实:机器运行时 + **本 (task,机器) 的任务数据位置** |
| ~~`paths.*`~~ | — | — | **env 已无 paths 段**:`fixed_pool`/`deploy_root`/`runs_base` 都是 **run 级事实**(本次 run 的输入与产物位置,在 run JSON);`benchmarks` 不声明,代码拼 `{deploy_root}/configs/tasks/<task>/benchmarks` |
| `resources` | 否 | map | 供大脑规划参考:`gpu_type` / `note` 等,自由字段 |
| `gateway` | 否 | map | 从这台机可达的 **LLM 端点(仅连接事实)**:`{base_url, key_file}`。**不含 model**——用哪个 model / 并发是**任务**的选择(→ `task.scoring.judge`) |
| ~~`train`~~ | — | — | **不声明,推导**:backend→env 菜单从 `tools.*.provides` 扫描得出(sft_env provides llamafactory ⇒ llamafactory→sft_env);dataos-train CLI 的位置同理(provides 含 `dataos-train` 者)。默认 backend 是代码常量,方法→env 查表在 dataos-train `env_for()` |
| `operations` | 是 | map | **★ 操作契约**:动词 → 命令模板(见 §4) |
| `discipline` | 是 | list[str] | **★ 硬约束**,一条一句(login 纪律、输出隔离、禁全盘扫描…) |

> **env 是 (task,机器) 部署单元**(名如 `PTB_H100_slurm`):它任务绑定,故该任务的数据路径(`fixed_pool` / `benchmarks`)就落在 **env.paths**(部署事实)。agent 自己的代码根(如 dataos src)仍是各 agent 的事,不进 env。

\* `access.host/user`:`kind: local` 时可省。

### operations 必含动词
`submit` `poll` `fetch` `cancel` `activate` 五个(见 §4 各自规范)。可加自定义动词,但这五个是 agent 依赖的最小集。

---

## 2. Task 契约 —— `task.yaml`

| 字段 | 必填 | 类型 | 说明 / 允许值 |
|---|---|---|---|
| `name` | 是 | str | 任务标识,与目录名一致(如 `posttrainbench`) |
| `inputs` | 是 | map | 给 agent 的输入(类型化引用) |
| `inputs.<名>` | — | map | 每项:`{from: <run.* 或 env.* 引用> 或 path:<路径>, kind: <类型>, one_of:[...](可选)}` |
| `objective` | 是 | str | 一句话目标(agent 要达成什么) |
| `constraints` | 是 | list[str] | 规则(只能用某数据、评测冻结、不得引入外部数据…) |
| `produce` | 是 | map | agent 要交付什么:`{kind: <产物类型,如 hf_model>, hint: <说明>}` |
| `scoring` | 是 | map | **★ 评分契约**(见 §5) |
| `assets` | 否 | map | 任务自带代码/数据的相对路径:`{<名>: <相对 task 目录的路径>}`(如 `lm_eval_bridge/`) |

### scoring 子字段
| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `scoring.evaluator` | 是 | path | 评测器入口(相对 task 目录) |
| `scoring.command` | 是 | str(模板) | 跑评测的命令模板,产出原生结果 |
| `scoring.judge` | 否 | map | judge 类 bench 的评分选择:`{model, workers}`(端点取自 `env.gateway`) |
| `scoring.judge_command` | 否 | str(模板) | judge 补分命令;用 `env.gateway.{base_url,key_file}` 连接 + `task.scoring.judge.{model,workers}` |
| `scoring.result` | 是 | map | **原生结果契约**:`{file: <模板路径>, field: <字段名>, range: [min,max]}` |
| `scoring.format_contract` | 否 | path/str | 指向 scorer 源码或其文档(供 agent 对齐格式) |
| `scoring.<其他>` | 否 | any | 允许任务附加**声明式**评分事实(如 curationbench 的 `aggregate` 归一化规则);组合器原样透传 |

---

## 3. 占位符命名空间(模板里 `{...}` 能用什么)—— 两阶段填充

| 命名空间 | 来源 | 何时填 | 例 |
|---|---|---|---|
| `{env.*}` | `env.yaml` 全字段 | **解析时**(compose.resolve) | `{env.access.host}` `{env.tools.eval_env.path}` `{env.paths.runs_base}` `{env.gateway.base_url}` |
| `{run.*}` | run-config | **解析时** | `{run.base_model}` `{run.bench}` `{run.resource.node}` `{run.run_id}`;`runs_base`(产物根,可选,默认 `<runtime_root>/runs`,相对/绝对皆可 —— compose 解析后注入 `{env.paths.runs_base}` 供模板引用)。任务数据在 env.paths,不在 run |
| `{ctx.*}` | 每个 experiment 运行时才知道 | **执行时**(sub-agent 自己填) | `{ctx.run_root}` `{ctx.env_path}` `{ctx.model_path}` `{ctx.eval_dir}` `{ctx.jobid}` `{ctx.script}` `{ctx.label}` `{ctx.remote}` `{ctx.local}` |
| `{task.*}` | `task.yaml` + 派生 | 解析时 | `{task.assets_root}`(= runtime 库里 task 目录的绝对根) |

> 模板引擎为 **jinja2**;占位符按点号路径取值(`{env.access.host}` 等)。

**规则**:`resolve()` 只填 `{env.*}/{run.*}/{task.*}`,保留 `{ctx.*}` 给 sub-agent。**`ctx.run_root` 由 agent 自选,唯一中立约束是必须落在 `{env.paths.runs_base}` 下**(其内部布局是各 agent 自己的事;DataOS 用 `{runs_base}/{run_id}/{study}/{exp}`,但不属本契约)。`ctx.*` 完整清单见 §6。

---

## 4. operations 动词规范

每个动词是一条命令模板;下表列其**用途 + 必用的 ctx 变量 + 期望输出**。

| 动词 | 用途 | 常用占位符 | 期望 |
|---|---|---|---|
| `activate` | 激活运行时(venv) | `{ctx.env_path}` | `source {ctx.env_path}/bin/activate`;更推荐直接 `{path}/bin/<tool>`,agent 按 stage 选 env_path |
| `submit` | 提交一个计算作业 | `{env.access.*}` `{run.slurm.*}` `{run.resource.*}` `{ctx.run_root}` `{ctx.script}` | 返回可解析出作业 id 的输出 |
| `poll` | 查作业状态 | `{env.access.*}` `{ctx.jobid}` | 输出含终态判定(COMPLETED/FAILED/…) |
| `fetch` | 取回产物到控制面 | `{env.access.*}` `{ctx.remote}` `{ctx.local}` | 文件落到 `{ctx.local}` |
| `cancel` | 取消作业 | `{ctx.jobid}` | 作业被取消 |

> `kind: local` 环境:`submit` 可以是直接跑命令、`poll` 恒 COMPLETED、`fetch` 为 `cp`。契约字段不变,模板内容随 kind 变。

---

## 5. resolved_runtime.json(compose.resolve 的产物形状)

```json
{ "runtime_ref": "v0.1.0",
  "environment": {"name","kind","access","tools","paths","gateway",
                  "operations": {"submit":"<env/run 已填,ctx 保留>","poll":"...","fetch":"...","cancel":"...","activate":"..."},
                  "discipline": ["...","..."]},
  "task": {"name","objective","inputs","constraints","produce",
           "scoring": {"command":"<已填>","judge_command":"...","result":{"file","field","range"},"format_contract":"..."},
           "assets_root": "<runtime_root>/tasks/<task>"},
  "run": {"run_id","base_model","bench","resource","slurm","eval_limit","budget_hours"} }
```

---

## 6. `ctx.*` 清单(执行时由 sub-agent 填)

| 变量 | 含义 |
|---|---|
| `ctx.run_root` | 本 experiment 的 scratch 根。**中立约束:必须落在 `env.paths.runs_base` 下**;内部布局由各 agent 自定(DataOS 用 `{runs_base}/{run_id}/{study}/{exp}`,但 study/exp 是 DataOS 概念,不属本契约) |
| `ctx.env_path` | `activate` 要激活的 tool 环境路径(取自 `env.tools.*` 中某个的 `path`,由 agent 按训练/评测阶段选) |
| `ctx.script` | sub-agent 写好的 sbatch 脚本路径 |
| `ctx.jobid` | submit 返回、poll/cancel 用的作业 id |
| `ctx.model_path` | 待评测/续训的模型目录(baseline 时 = base_model) |
| `ctx.eval_dir` | 评测产物目录(`scoring.result.file` 相对它);**建议**(非硬性)`{run_root}/eval/{bench}_limit{N}` |
| `ctx.label` | 评测 model-name 标签 |
| `ctx.remote` / `ctx.local` | fetch 的远端/本地路径 |

> **task 专属 ctx**:上表是环境操作依赖的公共集;task 的 scoring 命令可引入**本任务专属**的 `ctx.<name>`
> (如 curationbench 的 `ctx.submission_path` / `ctx.iter_run_dir`),须在该 task.yaml 的注释里说明取值来源。

---

## 7. 校验规则(compose.resolve 会检查)

- env:`name/kind/access/tools/operations/discipline` 齐全;`operations` 含五动词。(`runs_base` 不再要求 —— 由 run-config 提供,默认 `<runtime_root>/runs`)
- task:`name/inputs/objective/constraints/produce/scoring` 齐全;`scoring` 含 `evaluator/command/result`;`result` 含 `file/field/range`。
- 模板里出现的 `{env.*}/{run.*}` 必须能在 env.yaml/run-config 找到对应值,否则解析报错(契约不满足)。
- 未知顶层字段 → 警告(向前兼容,不报错)。

---

## 8. Task 目录布局(每个 task 的标准三件套)

```
tasks/<task>/
  task.yaml        总声明:inputs / objective / constraints / produce / scoring(见 §2)
  eval/            评测器:把一个【模型】打成原生分(每 task 各自实现;task.yaml.scoring.command 指向它)
  train/           可选。★ 目录存在 = 训练【冻结】(配方不可改);不存在 = 训练是 agent 的方法
```

- **三段 DATA → TRAIN → EVAL**:DATA 是 agent 杠杆;EVAL 永远"评一个模型 → 分数"(跑 task.yaml 的
  `scoring.command`,读 `result.file` 的 `result.field`);差别只在 **TRAIN 归属**。
- **`train/` 的语义(唯一信号)**:存在 → `compose.resolve` 置 `resolved.task.train = {frozen, command, ...}`
  (读 `train/train.yaml`);不存在 → `resolved.task.train = None`。大脑据此把 train stage 归 provided/ours。
- `train/train.yaml`(冻结时):`{frozen: true, command: <模板>, produces: <模型路径>}`;命令同样支持
  `{{env.*}}{{run.*}}` 解析 + `{ctx.*}` 保留。
- 多指标 bench 若需归一化(如把 8 个子分算成一个可比分),用 `scoring.aggregate` 声明规则,由消费方
  (大脑组装 report 时)按声明计算——不在 runtime 里常设 runner。

## 9. 加一个新环境 / 新任务

- 新环境:`environments/<env>/env.yaml` 按 §1 填全(operations 五动词 + discipline)。零业务代码。
- 新任务:建 `tasks/<task>/`,写 `task.yaml`(§2)+ `eval/`(评测器);训练冻结的话加 `train/`(§8)。
- 不必写 `compose.py`(通用组合器已在 `compose/`,对所有 task 通用)。
