# DataOS Runtime 契约(Environment / Task 抽象)

状态:**已实施;2026-07-29 起 DataOS-Runtime 全量迁入本仓库 `configs/`(environments/tasks/CONTRACT.md),外部仓库归档**。字段规范的当前真源是 `configs/CONTRACT.md`;本文件讲设计与时序。目标:把**散文 playbook** 升级为**带类型的声明式契约**,并明确「写什么 / 怎么拼 / 何时拼给 DataOS」。

> **当前实现要点(与早期草案的差异)**:① 组合器 `compose/`(resolve+render)在 runtime 库,agent 直接 import;② env 是 **(task,机器) 部署单元**(名如 `PTB_H100_slurm`),任务数据(fixed_pool/benchmarks)落 `env.paths`;③ 环境用 **uv venv**(sft/rl/eval,按 `{path}/bin/<tool>` 调,无 conda);④ 每 task = `task.yaml + eval/ + train/?`,`train/` 存在=训练冻结;⑤ EVAL 声明式(跑 `scoring.command` 读 `result.field`),无 common runner;⑥ posttrainbench 评测器扁平化为 `eval/{scripts,lmeval,models}`。

---

## 0. 为什么

现在 `environments/<env>/playbook.md` 和 `tasks/<task>/playbook.md` 是自由散文,sub-agent 自己读、自己领会。问题:没有 schema(不知道该覆盖什么)、没有拼装规则(env+task+run-config 怎么合成 agent 看到的东西)、没有明确时序。

改成:**runtime 声明结构化契约(`env.yaml` / `task.yaml`),DataOS 在 run 启动时把契约「解析 → 填充 → 渲染」成 sub-agent 指令。** schema 是唯一真源;`playbook.md` 由 schema 生成或降为人读伴生文档。

设计原则(沿用既有决策):
- **声明式 + 命令模板**;契约本身不是 Python API。
- **组合机制(拼接模板 + 拼接 py)放在 runtime 库**:runtime 不只暴露 schema,还附一份**通用组合器**(`compose/compose.py` + `compose/templates/*.j2`),任何 agent `resolve()+render()` 即可拿到「环境操作 + 任务」指令。这份 py 是**纯通用拼装逻辑,不含任何 agent 的私有方法**,所以 runtime 仍 agent-中立、可被别的 baseline agent 直接复用。模板引擎用 **jinja2**;调用方(如 DataOS)把 runtime 目录加进 `sys.path` **直接 import compose**(同进程,方案 A),故控制面 env 需装 jinja2。
- **environment 只讲「怎么操作这台机」,task 只讲「问题是什么 + 怎么打分」;都不含解法**(造数据/训练的方法是各 agent 私有,留各自的 agent 库)。
- runtime **agent-中立**:任何 agent 都能按同一契约 + 同一组合器驱动它。
- **字段规范**见 `configs/CONTRACT.md`(逐字段说明 env/task 能写什么);本文件讲设计与时序。

---

## 1. 三层模型

```
① 声明 Declaration   (runtime 库)     env.yaml + task.yaml —— 结构化 schema(权威)
② 解析 Resolution    (DataOS 大脑)     run-config 绑定(env,task,ref)+ per-run
                                       → 填充模板占位符 → resolved_runtime.json
③ 渲染 Rendering      (DataOS 大脑)     resolved_runtime → sub-agent 提示词的
                                       「环境操作」段 + 「任务」段
```

sub-agent 最终提示词 = **[大脑的方法段(算子/LLaMA-Factory,DataOS 私有)] + [渲染出的 环境操作 + 任务定义] + [大脑的 report.json 数据契约]**。中间那段由 **runtime 自带的组合器**从结构化契约渲染,不再是"读个 md 自己领会";两头的私有段由各 agent 自己拼。

### runtime 库布局(含组合机制)
```
configs/  (迁入后)
  environments/<env>/env.yaml          # 声明:环境契约
  tasks/<task>/task.yaml               # 声明:任务契约
  tasks/<task>/eval/                   # 任务自带的评测器(把模型打成分);train/ 可选(存在=冻结)
  compose/
    compose.py                         # ★ 通用组合器:resolve(run_config) + render(resolved)
    templates/environment.j2         # ★ 「环境操作」段的提示词模板
    templates/task.j2                # ★ 「任务」段的提示词模板
  CONTRACT.md                          # ★ 字段规范(env/task 每个字段能写什么)
  README.md
```
任何 agent(DataOS 或别的 baseline)统一这样用:
```python
from compose import resolve, render
resolved = resolve(run_config)                 # 读 env.yaml+task.yaml@ref,填 {env.*}{run.*}
env_ops, task_def = render(resolved)            # 两段可直接放进提示词的文本
# 之后各 agent 拼自己的:数据构建/训练方法、结果如何消费(report.json / 自己的记录)
```

---

## 2. Environment 契约 —— `environments/<env>/env.yaml`

声明**一组能力/操作 + 事实 + 硬约束**。字段:

| 段 | 必填 | 内容 |
|---|---|---|
| `name` `kind` | 是 | 标识 + 类型(`slurm`/`k8s`/`local`) |
| `access` | 是 | 怎么到达:`host/user/port/key_ref`(私钥路径,秘钥本身不入库) |
| `tools` | 是 | 有哪些可用运行时(声明,不强制用):每项 `{path, provides:[...]}` |
| `paths` | 是 | 本部署的路径事实 `fixed_pool`/`deploy_root`/`runs_base`/`benchmarks`(唯一声明处;run JSON 不得重复) |
| `resources` | 否 | 供大脑规划参考:`gpu_type` 等 |
| `gateway` | 否 | 从这台机可达的 LLM 端点(仅连接):`base_url/key_file`;model/并发在 task.scoring.judge |
| `operations` | 是 | **★ 操作契约**:生命周期动词 → 参数化命令模板(见 §4) |
| `discipline` | 是 | **★ 硬约束**:字符串列表(login 纪律、输出隔离…) |

`operations` 必含的动词:`submit` / `poll` / `fetch` / `cancel` / `activate`。

**示例(PTB_H100_slurm,uv 环境):**
```yaml
name: PTB_H100_slurm                          # env = (task,机器) 部署单元
kind: slurm
access:   {host: <login-node>, user: <you>, port: 22, key_ref: ~/.ssh/<key>}
tools:                                         # uv venv;按 {path}/bin/<tool> 调,无 conda
  sft_env:  {path: <root>/environments/PTB_H100_slurm/envs/sft/.venv,  provides: [llamafactory, torch, flash-attn]}
  rl_env:   {path: <root>/environments/PTB_H100_slurm/envs/rl/.venv,   provides: [verl, vllm, ray]}
  eval_env: {path: <root>/environments/PTB_H100_slurm/envs/eval/.venv, provides: [vllm, lm-eval]}
paths:                                         # 机器事实 + 本部署的任务数据(env 任务绑定)
  runs_base: <deploy_root>_runs
  deploy_root: /path/on/the/exec/host/dataos
  fixed_pool: /data2/.../posttrainbench_fixed_pool_v2
  benchmarks: /data/.../posttrainbench_benchmarks
gateway:  {base_url: https://api.chatanywhere.tech/v1, key_file: ...}   # 仅连接;model/并发在 task.scoring.judge
operations:
  activate: "source {ctx.env_path}/bin/activate"
  submit:   "ssh -i {env.access.key_ref} -p {env.access.port} {env.access.user}@{env.access.host}
             'sbatch --partition={env.slurm.partition} --qos={env.slurm.qos} --gres=gpu:{run.resource.gpus}
              --nodelist={run.resource.node} --cpus-per-task={run.resource.cpus}
              --chdir={ctx.run_root} --output={ctx.run_root}/logs/slurm-%j.out {ctx.script}'"
  poll:     "ssh ... {env.access.user}@{env.access.host}
             'sacct -j {ctx.jobid} --format=State --noheader -n | awk \"{{$1=$1;print $1}}\"'"
  fetch:    "scp -i {env.access.key_ref} -P {env.access.port}
             {env.access.user}@{env.access.host}:{ctx.remote} {ctx.local}"
  cancel:   "ssh ... 'scancel {ctx.jobid}'"
discipline:
  - "login 节点只 submit/poll(每次 sleep ≥30s)+ 有界联网步骤;一切重活进 sbatch"
  - "输出隔离:一切产物落 {ctx.run_root};sbatch 钉 --chdir/--output;cache/TMPDIR 重定向进 run_root"
  - "绝不递归/全盘扫描 /data"
```

---

## 3. Task 契约 —— `tasks/<task>/task.yaml`

声明**问题定义 + 评分契约 + 资产**。字段:

| 段 | 必填 | 内容 |
|---|---|---|
| `name` | 是 | 标识 |
| `inputs` | 是 | 给 agent 的东西(类型化引用):base_model / data_pool / target 等 |
| `objective` | 是 | 一句话目标 |
| `constraints` | 是 | 规则列表(只能用池内数据、评测器冻结…) |
| `produce` | 是 | agent 要交付什么:`{kind: hf_model}` |
| `scoring` | 是 | **★ 评分契约**:evaluator 命令模板 + 原生结果契约 + 格式契约指针 |
| `assets` | 否 | 任务自带代码/数据的相对路径(eval bridge 等) |

**示例(posttrainbench):**
```yaml
name: posttrainbench
inputs:
  base_model:   {from: run.base_model, kind: hf_model}
  data_pool:    {from: env.paths.fixed_pool, kind: fixed-pool}   # env 任务绑定,数据在 env.paths
  target_bench: {from: run.bench, one_of: [gsm8k, aime2025, humaneval, bfcl, gpqamain, healthbench, arenahardwriting]}
objective: "只用 data_pool 里的数据,把 base_model 在 target_bench 上的分数做高"
constraints:
  - "只允许用 data_pool 内的 source,不得引入外部数据"
  - "评测器冻结;base 与产出模型走同一套评测"
produce: {kind: hf_model, hint: "一个完整 HF 模型目录"}
scoring:
  evaluator: "eval/scripts/run_lmeval_eval.py"
  command: "{env.tools.eval_env.path}/bin/python {task.assets_root}/eval/scripts/run_lmeval_eval.py
            --task {run.bench} --model-path {ctx.model_path} --model-name {ctx.label}
            --output-dir {ctx.eval_dir} --limit {run.eval_limit}"
  judge: {model: qwen3.5-plus, workers: 32}   # 本任务的 judge 选择(端点用 env.gateway)
  judge_command: "... {{ env.gateway.base_url }} ... {{ task.scoring.judge.model }} ..."
  result: {file: "{ctx.eval_dir}/summary.json", field: "score", range: [0, 100]}
  format_contract: "eval/lmeval/utils.py"
assets:
  eval: "eval/"
```

---

## 4. 模板与占位符(怎么填)—— **两阶段填充**

命令模板里的 `{...}` 分两个命名空间、两个时机:

| 命名空间 | 来源 | 何时填 |
|---|---|---|
| `{env.*}` | env.yaml 的字段 | **解析时**(run 启动) |
| `{run.*}` | run-config(base_model/bench/resource/slurm/paths/budget…) | **解析时** |
| `{ctx.*}` | 每个 experiment 运行时才知道的量:`run_root`/`env_path`/`model_path`/`eval_dir`/`jobid`/`script`/`label`/`remote`/`local` | **执行时**(sub-agent 自己填) |

所以解析器只填 `env.*` 和 `run.*`,产出**半解析模板**;`ctx.*` 保留给 sub-agent 在真正提交/轮询/评测时补上。**`ctx.run_root` 由 agent 自己选,中立契约只约束它必须落在 `env.paths.runs_base` 下**(fetch/输出隔离才成立);内部布局是各 agent 的事——DataOS 自己用 `{runs_base}/{run_id}/{study}/{exp}`(study/exp 是 DataOS 实验树概念,写在大脑侧,**不进本契约**)。`ctx.*` 完整清单见 `CONTRACT.md §6`。

> 转义/注入:模板是 shell 命令,占位符值来自受控的 env.yaml/run-config,不含用户自由文本;渲染时对含空格/特殊字符的值按需加引号。

---

## 5. 组合(绑定点 = run-config)

```json
{ "environment": "h100_slurm", "task": "posttrainbench", "runtime_ref": "v0.1.0",
  "run_id": "...", "base_model": "...", "bench": "gsm8k",
  "resource": {"node":"gpu04","gpus":1,"cpus":8}, "eval_limit": -1, "budget_hours": 10,
  "fixed_pool": "/data2/.../posttrainbench_fixed_pool_v2", "benchmarks": "/data/.../posttrainbench_benchmarks" }
```
`environment`+`task`+`runtime_ref` 选定 runtime 里的两份 schema;其余是 per-run 事实,喂给 `{run.*}`。

---

## 6. 解析(Resolution)—— runtime 的 `compose.resolve()`

`compose.resolve(run_config) -> resolved_runtime.json`(住在 **runtime 库** `compose/compose.py`,DataOS 与其他 agent 都调它):
1. 按 `runtime_ref` checkout / 定位本地 runtime 库。
2. 读 `environments/{environment}/env.yaml` + `tasks/{task}/task.yaml`。
3. 用 `{env.*}`+`{run.*}` 填 `operations` / `scoring.command` 等模板(保留 `{ctx.*}`)。
4. 产出 `resolved_runtime.json`:
```json
{ "runtime_ref": "v0.1.0",
  "environment": {"name","kind","access","tools","paths","gateway",
                  "operations": {"submit":"<半解析>","poll":"...","fetch":"...","activate":"..."},
                  "discipline": ["...","..."]},
  "task": {"name","objective","inputs","constraints","produce",
           "scoring": {"command":"<半解析>","result":{...},"format_contract":"...","judge_command":"..."},
           "assets_root": "<configs>/tasks/posttrainbench"},
  "run": {"run_id","base_model","bench","resource","budget_hours","eval_limit"} }
```
校验:必填段齐全、operations 五个动词都在、scoring.result 完整;缺则报错(契约不满足)。

---

## 7. 渲染(Rendering)—— runtime 的 `compose.render()`

`compose.render(resolved_runtime) -> (env_ops, task_def)`(住在 runtime `compose/compose.py`,用 `compose/templates/*.j2`),产出 sub-agent 提示词里的两段:
- **ENVIRONMENT OPS**(`templates/environment.j2`):逐条列 `operations`(submit/poll/fetch/activate 的半解析命令)+ `discipline` 约束 + `ctx.*` 说明。
- **TASK**(`templates/task.j2`):`objective` + `inputs` + `constraints` + `produce` + `scoring`(评测命令 + result 契约 + format_contract 指针)。

**调用方 agent(如 DataOS)** 再拼上**自己的**两段(不来自 runtime):
- **METHOD**(DataOS 私有解法):用算子造 SFT 数据、LLaMA-Factory 训练、格式对齐、operator 沉淀。
- **REPORT CONTRACT**(数据契约):把 scoring.result 的原生分数组装成 report.json 的形状。

拼接顺序固定,渲染确定性可测。

---

## 8. 时序(什么时候拼给 DataOS)

- **run 启动时**(`loop.py run_auto` 开头):DataOS 调 runtime 的 `compose.resolve(run_config)` 一次 → 得 `resolved_runtime`(写进 run 目录做溯源)。
- **每个 experiment**:`compose.render(resolved_runtime)` 得两段 + 大脑 METHOD/REPORT 段 + per-exp 的 `ctx.*`(run_root/model_path/eval_dir)→ sub-agent 提示词。
- **`runtime_ref` 变了**才重新 checkout + resolve;契约破坏体现在版本号。
- DataOS 怎么调 runtime 的 compose(**方案 A,直接 import**):控制面 checkout 了 runtime 库,把其目录加进 `sys.path`,`import compose` 后 `compose.resolve()/render()` 当函数调(同进程)。compose 用 **jinja2** 渲染模板 → 控制面 Python env 需装 jinja2(其他 Python agent 复用时同装)。

---

## 8.5 compose 已知边界与注意(2026-07-21 审计)

无深层设计缺陷;以下是使用/维护时要清楚的语义与脆弱点:

1. **`runtime_ref` 已是遗留字段**:configs 迁入本仓库后,版本就是本仓库的 git 版本,`resolve(run_config, configs_root)` 读的是仓库内 `configs/`;`runtime_ref` 仅作为透传元数据保留(恒 `"main"`),不再承担 pin 语义。
2. **渲染范围不是整棵树**:只有 `operations` / `discipline` / `scoring.command` / `judge_command` / `scoring.result` 会被 jinja 填 `{{...}}`;`objective` / `constraints` / `inputs` / `produce` **原样透传**(它们是描述性文本)。→ task 作者:`{{...}}` 只写在会被渲染的字段里;描述字段里的引用(如 `data_pool.from: env.paths.fixed_pool`)是**字面标签**,不是模板。
3. **缺失可选字段的守卫**:`format_contract` 可选,`task.j2` 已加 `{% if %}` 守卫(否则缺失时 render 会崩)——新增可选字段进模板时记得同样守卫。
4. **缺 `{{run.X}}` 的报错**:命令模板用 StrictUndefined,run-config 缺键会抛 jinja `UndefinedError`(点名该属性),信息够定位但不算友好。
5. **`ctx.*` 只支持单层**:`_Ctx` 代理把 `{{ctx.foo}}` 变字面量 `{ctx.foo}`;`{{ctx.foo.bar}}` 这种嵌套会失败(当前无此用法)。
6. **`resolved` 是 JSON 安全的**(全是 dict/str),故 CLI 可 resolve→写盘→render 分两步跑。

---

## 9. playbook.md 策略

`env.yaml`/`task.yaml` 是**唯一真源**。`playbook.md`:
- 要么由 schema **自动生成**(给人读的渲染版,和渲染给 agent 的同源);
- 要么直接删除,人读需求靠本设计文档 + schema 注释。

不再手写散文 playbook,杜绝两处不一致。

---

## 10. 迁移计划

- [x] **C1 定稿本设计文档**(本文件)+ **CONTRACT.md 字段规范**(现 `configs/CONTRACT.md`)。
- [x] **C2 runtime 库 · 声明**:给 `h100_slurm/env.yaml` 补 `operations`/`discipline`/`tools`;为 `posttrainbench` 写 `task.yaml`(从现有两份 playbook 抽结构)。
- [x] **C3 runtime 库 · 组合器**:实现 `compose/compose.py`(`resolve` 读 schema+填模板+校验;`render` 用 jinja2 渲染 `templates/*.j2` 出两段)+ 模板。删/降级 playbook.md(改由 render 产出人读版)。
- [x] **C4 DataOS · 接线**:sub-agent 提示词改为「调 runtime.compose 得两段 + 大脑 METHOD/REPORT 段」,替换硬写的 "read playbook.md"。
- [x] **C5 端到端**(2026-07-19 与 C6 合并验证:baseline 41.0 二分点过;假 baseline agent 零依赖复用 ✓):选 gsm8k 起一次 run,对拍现结果(baseline≈41),确认渲染出的指令与现在等价或更好。同时用一个"假 baseline agent"直接调 compose 验证中立可复用。
- [x] **C6 接入 DataOS-train(按 §12 依赖规则;runtime 侧零改动)**(2026-07-19 验证:4 个训练实验经 dataos-train,best 86.0 > 旧手写路径 80;RL/verl 与在线算子归一延后):
  - C6a(train 侧):删除内置的 posttrainbench reward(任务代码出库,`verl/data.py` 的 bridge import 一并改注入);`Reward` 由注册表 ref 改为**注入的文件路径**(runtime-owned);**reward 文件协议**(`compute_score(data_source, solution_str, ground_truth)` 签名)写进 train 的 schema 文档;`validate()` 预检:reward 路径存在且含 compute_score、online op 名可解析;hook 按名发现(entry-point/注册表)。
  - C6b(brain 侧):resolved 契约 → TrainSpec 适配器(env/cluster 事实注入,单一事实源);TRAIN stage 的 `invoke` 换成 dataos-train validate/build → env.operations.submit/poll → collect;**RL 方法时**,METHOD 指引 sub-agent 读 `format_contract` 自己写 `$RUN_ROOT/reward.py`(按宿主协议)+ 自检(几对已知 (solution, ground_truth) 对拍)后把路径填进 TrainSpec;dataos-train pip 进 H100 train env。
  - C6c(验证):SFT 经 dataos-train 端到端对拍;RL(GRPO)冒烟待 verl env 建好;在线算子机制归一(dataos_operator 出逻辑、train hook 寄宿)最后做。

---

## 11. 已定的决策

- ✅ **契约形式**:声明式 YAML + 命令模板(非 Python API)。
- ✅ **模板引擎**:jinja2。
- ✅ **调用方式**:方案 A,DataOS 把 runtime 目录加进 `sys.path` 直接 `import compose`(同进程);控制面 env 装 jinja2。
- ✅ **judge 补分**:放 `task.scoring.judge_command`(属评分的一部分),不设成 env operation。
- ✅ **playbook.md**:schema 权威,md 由 render 生成人读版或删除。
- ✅ **run_root**:agent 自选、中立契约只约束在 `env.paths.runs_base` 下;DataOS 的 `{runs_base}/{run_id}/{study}/{exp}` 布局归大脑,不进契约。
- ✅ **ctx.\* 清单**:见 CONTRACT.md §6(run_root/env_path/script/jobid/model_path/eval_dir/label/remote/local)。

## 12. 跨阶段接缝与依赖规则(data × train × eval)

三阶段在**实现层**不可能完全解耦——在线算子要寄宿在训练循环里、RL 的 reward 要调 eval 的 verifier、
造数据要对齐 scorer 的解析形状。耦合集中在三条边,**全部按「运行时注入」组织,仓库间保持零
build-time 依赖**:

| 边 | 载体 | 绑定方式 | 接口归属 |
|---|---|---|---|
| eval → data(格式契约) | scorer 源码(`task.yaml scoring.format_contract` 指针) | **只读文件**,无代码绑定 | task 契约 |
| eval → train(RL reward) | reward 文件(含 `compute_score`)——**由 sub-agent 运行时自己写**:读 scorer 源码(`format_contract`)派生,落在 run_root | **文件路径注入**(verl `custom_reward_function.path` 原生按路径加载,无 import);路径由大脑填进 TrainSpec | **协议**(函数名/签名)归 train/宿主;**内容**归 agent 方法——任务不 ship reward,对 RL 零知识。"reward==eval" 是 DataOS 方法的自我要求(可作实验轴),不是库保证 |
| data → train(在线算子寄宿) | 数值逻辑(torch-free,属 dataos_operator)+ 宿主胶水(属 train 的 `ext/`,唯一碰 LF 内部处) | **名字+参数注入**(`TrainSpec.online_ops`);宿主运行时按名解析(entry-point/注册表) | **hook 协议由 train(宿主)拥有**,按约定成文(duck-typed:`hook_point` 属性 + `__call__` 签名),不共享 base class |

### 七条规则

1. **零 build-time 依赖**:`dataos_operator` / `DataOS-train` 互不 import;所有跨阶段边 = 路径/名字 + 约定协议(configs 已入库,不再存在 runtime 依赖)。
2. **接口归属 = 宿主**:代码在谁的进程里执行,协议就归谁(train 拥有 hook 协议;eval/task 拥有 result 与 verifier 形状)。
3. **组装点唯一 = 大脑**:从 resolved 契约取 verifier 路径、按 plan 填 online_ops、注入 env/cluster 事实拼 TrainSpec——绑定集中在一处(composition root),库之间互不知晓。
4. **契约流逆流水线**:执行顺序 data→train→eval,契约方向 eval→train→data(冻结 scorer 是不动点,上游一切对齐它)。所有逆向边**只读**——尺子可读不可弯。
5. **错误前置**:运行时绑定的破绽(路径不存在 / op 名打错 / 协议漂移)由 train 的 `validate()` 预检 + `runtime_ref`/Capabilities 版本化兜住,不留到 GPU 作业里炸。
6. **部署共址 ≠ 代码耦合**:dataos_operator 与 dataos-train 需同装进 train conda env(作业内执行 convert/entry/算子)——这是环境事实,记在 env.yaml `tools.provides`。
7. **在线算子机制归一**:train 的 hook seam 是**唯一**寄宿机制;dataos_operator 只保留数值逻辑,弃用其自带的 LF 集成(`dataos_enable` 路线)。

由此的职责收缩:**train = validate/build/collect(方法层,任务无关、环境无关)**;提交/轮询/激活走
env 契约的 operations(train 自带的 slurm launcher 在 DataOS 体系内不使用);**任务不 ship 任何
方法产物**——reward 与数据对齐脚本同类,都是 agent 读冻结 scorer 后派生的方法产物(风险自担,
用类似 format pre-check 的自检兜底:拿几对已知 (solution, ground_truth) 验证自己写的 reward 与
评测判定一致)。

## 13. 仍需注意 / 风险

- **多环境/多任务通用性**:schema 要能容纳非 slurm 环境(k8s/local)和非 PTB 任务而不改 DataOS——agent-中立的试金石(C5 用"假 baseline agent"验)。
- **渲染 vs 自愈自由度**:渲染出的命令是"模板/建议",要明确允许 sub-agent 按需做兼容性微调,别把它当死命令。
- **jinja2 依赖**:控制面 + 任何复用方的 Python env 都要有 jinja2(轻依赖,可接受)。
