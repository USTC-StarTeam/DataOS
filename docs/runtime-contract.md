# DataOS Runtime 契约(Environment / Task 抽象)

状态:**设计定稿中(待评审)**。目标:把 DataOS-Runtime 里现在的**散文 playbook** 升级为**带类型的声明式契约**,并明确「写什么 / 怎么拼 / 何时拼给 DataOS」。

---

## 0. 为什么

现在 `environments/<env>/playbook.md` 和 `tasks/<task>/playbook.md` 是自由散文,sub-agent 自己读、自己领会。问题:没有 schema(不知道该覆盖什么)、没有拼装规则(env+task+run-config 怎么合成 agent 看到的东西)、没有明确时序。

改成:**runtime 声明结构化契约(`env.yaml` / `task.yaml`),DataOS 在 run 启动时把契约「解析 → 填充 → 渲染」成 sub-agent 指令。** schema 是唯一真源;`playbook.md` 由 schema 生成或降为人读伴生文档。

设计原则(沿用既有决策):
- **声明式 + 命令模板**;契约本身不是 Python API。
- **组合机制(拼接模板 + 拼接 py)放在 runtime 库**:runtime 不只暴露 schema,还附一份**通用组合器**(`compose/compose.py` + `compose/templates/*.j2`),任何 agent `resolve()+render()` 即可拿到「环境操作 + 任务」指令。这份 py 是**纯通用拼装逻辑,不含任何 agent 的私有方法**,所以 runtime 仍 agent-中立、可被别的 baseline agent 直接复用。模板引擎用 **jinja2**;调用方(如 DataOS)把 runtime 目录加进 `sys.path` **直接 import compose**(同进程,方案 A),故控制面 env 需装 jinja2。
- **environment 只讲「怎么操作这台机」,task 只讲「问题是什么 + 怎么打分」;都不含解法**(造数据/训练的方法是各 agent 私有,留各自的 agent 库)。
- runtime **agent-中立**:任何 agent 都能按同一契约 + 同一组合器驱动它。
- **字段规范**见 `DataOS-Runtime/CONTRACT.md`(逐字段说明 env/task 能写什么);本文件讲设计与时序。

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
DataOS-Runtime/
  environments/<env>/env.yaml          # 声明:环境契约
  tasks/<task>/task.yaml               # 声明:任务契约
  tasks/<task>/lm_eval_bridge/         # 任务自带的评测器代码(有依赖)
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
| `paths` | 是 | **只放机器文件系统事实**:`runs_base`(必填)、`runtime_root`;任务数据(fixed_pool/benchmarks)走 run-config |
| `resources` | 否 | 供大脑规划参考:`gpu_type` 等 |
| `gateway` | 否 | 从这台机可达的 LLM 端点(仅连接):`base_url/key_file`;model/并发在 task.scoring.judge |
| `operations` | 是 | **★ 操作契约**:生命周期动词 → 参数化命令模板(见 §4) |
| `discipline` | 是 | **★ 硬约束**:字符串列表(login 纪律、输出隔离…) |

`operations` 必含的动词:`submit` / `poll` / `fetch` / `cancel` / `activate`。

**示例(h100_slurm):**
```yaml
name: h100_slurm
kind: slurm
access:   {host: 114.214.215.159, user: hcgu, port: 22, key_ref: ~/.ssh/ghc_miyao}
tools:
  train_env: {path: /data/hcgu/.conda/envs/Llama-op020, provides: [llamafactory]}
  eval_env:  {path: /data/hcgu/.conda/envs/lm-eval,    provides: [lm-eval, vllm]}
  conda_sh:  /data/software/miniconda3/etc/profile.d/conda.sh
paths:    {runs_base: /data/hcgu/users/tjh/dataos_agentver_runs, runtime_root: /data/hcgu/users/tjh/dataos_runtime}  # 只放机器文件系统事实
gateway:  {base_url: https://api.chatanywhere.tech/v1, key_file: ...}   # 仅连接事实;model/并发在 task.scoring.judge
operations:
  activate: "source {env.tools.conda_sh} && conda activate {ctx.env_path}"
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
  data_pool:    {from: run.fixed_pool, kind: fixed-pool}   # 路径由 run-config 给,不绑死某台机
  target_bench: {from: run.bench, one_of: [gsm8k, aime2025, humaneval, bfcl, gpqamain, healthbench, arenahardwriting]}
objective: "只用 data_pool 里的数据,把 base_model 在 target_bench 上的分数做高"
constraints:
  - "只允许用 data_pool 内的 source,不得引入外部数据"
  - "评测器冻结;base 与产出模型走同一套评测"
produce: {kind: hf_model, hint: "一个完整 HF 模型目录"}
scoring:
  evaluator: "assets/lm_eval_bridge/scripts/run_lmeval_eval.py"
  command: "python {task.assets_root}/lm_eval_bridge/scripts/run_lmeval_eval.py
            --task {run.bench} --model-path {ctx.model_path} --model-name {ctx.label}
            --output-dir {ctx.eval_dir} --limit {run.eval_limit}"
  judge: {model: qwen3.5-plus, workers: 32}   # 本任务的 judge 选择(端点用 env.gateway)
  judge_command: "... {{ env.gateway.base_url }} ... {{ task.scoring.judge.model }} ..."
  result: {file: "{ctx.eval_dir}/summary.json", field: "score", range: [0, 100]}
  format_contract: "assets/lm_eval_bridge/ptb/eval_tasks/posttrainbench/utils.py"
assets:
  lm_eval_bridge: "lm_eval_bridge/"
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
           "assets_root": "<runtime_root>/tasks/posttrainbench"},
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

## 9. playbook.md 策略

`env.yaml`/`task.yaml` 是**唯一真源**。`playbook.md`:
- 要么由 schema **自动生成**(给人读的渲染版,和渲染给 agent 的同源);
- 要么直接删除,人读需求靠本设计文档 + schema 注释。

不再手写散文 playbook,杜绝两处不一致。

---

## 10. 迁移计划

- [ ] **C1 定稿本设计文档**(本文件)+ **`DataOS-Runtime/CONTRACT.md` 字段规范**。
- [ ] **C2 runtime 库 · 声明**:给 `h100_slurm/env.yaml` 补 `operations`/`discipline`/`tools`;为 `posttrainbench` 写 `task.yaml`(从现有两份 playbook 抽结构)。
- [ ] **C3 runtime 库 · 组合器**:实现 `compose/compose.py`(`resolve` 读 schema+填模板+校验;`render` 用 jinja2 渲染 `templates/*.j2` 出两段)+ 模板。删/降级 playbook.md(改由 render 产出人读版)。
- [ ] **C4 DataOS · 接线**:sub-agent 提示词改为「调 runtime.compose 得两段 + 大脑 METHOD/REPORT 段」,替换硬写的 "read playbook.md"。
- [ ] **C5 端到端**:选 gsm8k 起一次 run,对拍现结果(baseline≈41),确认渲染出的指令与现在等价或更好。同时用一个"假 baseline agent"直接调 compose 验证中立可复用。

---

## 11. 已定的决策

- ✅ **契约形式**:声明式 YAML + 命令模板(非 Python API)。
- ✅ **模板引擎**:jinja2。
- ✅ **调用方式**:方案 A,DataOS 把 runtime 目录加进 `sys.path` 直接 `import compose`(同进程);控制面 env 装 jinja2。
- ✅ **judge 补分**:放 `task.scoring.judge_command`(属评分的一部分),不设成 env operation。
- ✅ **playbook.md**:schema 权威,md 由 render 生成人读版或删除。
- ✅ **run_root**:agent 自选、中立契约只约束在 `env.paths.runs_base` 下;DataOS 的 `{runs_base}/{run_id}/{study}/{exp}` 布局归大脑,不进契约。
- ✅ **ctx.\* 清单**:见 CONTRACT.md §6(run_root/env_path/script/jobid/model_path/eval_dir/label/remote/local)。

## 12. 仍需注意 / 风险

- **多环境/多任务通用性**:schema 要能容纳非 slurm 环境(k8s/local)和非 PTB 任务而不改 DataOS——agent-中立的试金石(C5 用"假 baseline agent"验)。
- **渲染 vs 自愈自由度**:渲染出的命令是"模板/建议",要明确允许 sub-agent 按需做兼容性微调,别把它当死命令。
- **jinja2 依赖**:控制面 + 任何复用方的 Python env 都要有 jinja2(轻依赖,可接受)。
