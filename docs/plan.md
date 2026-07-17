# DataOS 拆分执行计划 (Brain / Runtime)

状态:**P1–P6 全部完成(2026-07-17)**。把执行侧(多环境 × 多任务)从 `DataOS` 拆到 `DataOS-Runtime`。
额外收尾:按用户指示把 runtime 做成 **agent-中立**(别的 baseline agent 也能用同一底座)——report.json/RECON/
sub-agent 框架抽回大脑,runtime 只暴露操作契约 + 原生评分契约。gsm8k 端到端验证在新结构上出分
(baseline 41.0 ≈ 已知 40.6)。

---

## 0. 已锁定的决策(原开放问题的答案)

1. **操作接口的载体 = PLAYBOOK + 配置(声明式),不是 Python API。**
   环境适配不写 `.py`,而是 `playbook.md`(命令模板)+ `env.yaml`(资源/路径)。sub-agent(Claude Code)读它、用 shell 自己干。加新环境 = 写一份新 playbook+yaml,零代码。
2. **PLAYBOOK 拆成"环境"和"任务"两半。**
   环境部分(ssh/sbatch/sacct/scp 怎么在这台集群上提交/轮询/取回)进 `environments/<env>/playbook.md`;任务部分(build→train→eval 的跑法 + 原生结果 → report.json 的组装)进 `tasks/<task>/playbook.md`。
3. **sub-agent 从 `DataOS-Runtime` 库读 PLAYBOOK**(不再从大脑库读)。控制面会 checkout 一份 runtime 库。
4. **`core/` 整个留在大脑;执行侧对 `dataos.core` 零依赖。**
   实测执行侧只 import `dataos_operator`(独立包),没 import `dataos.core`。数据契约(schema)留大脑,执行侧只产出原生结果 + 按任务 playbook 组装 report.json。

---

## 1. 目标结构

```
DataOS (Brain 库) ── 环境/任务无关
  src/dataos/agent/     loop · backend · llm · planner · reviewer · config · subagent_executor
  src/dataos/core/      experiment · study · project · workspace · report(契约 schema) · ...
  src/dataos/prompt/    通用 agent prompts (main_agent, strategy, structured_*, subagent 骨架)
  configs/runs/         run-config(选 environment + task)
  ── 持有:全部数据契约 schema;读 report.json 时做校验 ──

DataOS-Runtime (执行库) ── 可插拔、对大脑 schema 零依赖
  environments/
    h100_slurm/
      env.yaml          host/partition/conda-envs/paths/resources 声明
      playbook.md       环境操作:ssh/sbatch/sacct/scp 命令模板
  tasks/
    posttrainbench/
      playbook.md       任务:build→train→eval 跑法 + 原生结果 → report.json 组装
      lm_eval_bridge/   eval bridge + scorers(从 DataOS/resources/code/posttrainbench 迁)
      operators/        custom_operators(从 DataOS/resources/custom_operators 迁)
  README.md
```

**绑定点 = run-config**:`environment: h100_slurm` + `task: posttrainbench` → sub-agent 读对应的两份 playbook + env.yaml。

---

## 2. 契约(大脑 ↔ 执行侧,唯一边界)

- **数据契约(留大脑)**:`report.json` / `recon.json` / `run-config` 的 JSON 形状。大脑读时校验;执行侧按任务 playbook 产出符合形状的 JSON,不 import 大脑。
- **操作契约(执行侧 conform)**:一份 playbook,能让 sub-agent 完成 submit / poll / fetch / 知道有哪些资源。不是 API,是命令模板 + env.yaml。
- **版本 pin**:run-config 记 `runtime_ref`(runtime 库的 tag/commit);控制面按它 checkout runtime,契约破坏体现在版本号。

---

## 3. "什么迁到哪"清单

| 现在位置(DataOS) | 迁到(DataOS-Runtime) | 备注 |
|---|---|---|
| `resources/code/posttrainbench/lm_eval_bridge/` | `tasks/posttrainbench/lm_eval_bridge/` | 纯平移 |
| `resources/code/posttrainbench/PLAYBOOK.md` | **拆**:环境部分→`environments/h100_slurm/playbook.md`;任务部分→`tasks/posttrainbench/playbook.md` | 拆分,最需小心 |
| `resources/custom_operators/` | `tasks/posttrainbench/operators/`(或 `environments` 共用) | 平移;`DATAOS_OPERATOR_PLUGINS` 指过去 |
| run_config `DEFAULT_CLUSTER`/paths | `environments/h100_slurm/env.yaml` | 硬编码 → 环境档案 |

**留在大脑不动**:`src/dataos/agent/*`、`src/dataos/core/*`、`src/dataos/prompt/*`(通用部分)、`configs/runs/*`。

---

## 4. 执行阶段(逐步、每步可验证、可回滚到 `v0.3.0-opus-baseline`)

- [x] **P1 — 搭骨架**:clone `DataOS-Runtime`,建 `environments/h100_slurm/` + `tasks/posttrainbench/` 目录 + README + 空 env.yaml/playbook.md 占位。提交。
- [x] **P2 — 平移任务代码**:把 `lm_eval_bridge/` + `custom_operators/` 原样迁进 `tasks/posttrainbench/`(不改逻辑)。提交。
- [x] **P3 — 拆 PLAYBOOK**:把现有 `PLAYBOOK.md` 逐段归类,拆成 `environments/h100_slurm/playbook.md`(集群操作)+ `tasks/posttrainbench/playbook.md`(任务流程 + report 组装)。写 `env.yaml`(从 DEFAULT_CLUSTER/run-config 抽)。提交。
- [x] **P4 — 大脑接线**:sub-agent 提示词改为"读 `{runtime}/environments/{env}/playbook.md` + `{runtime}/tasks/{task}/playbook.md`";run_config 增加 `environment`/`task`/`runtime_ref` 字段,`DEFAULT_CLUSTER` 改为从 env.yaml 读。控制面 checkout runtime 库的机制。import 冒烟。
- [x] **P5 — 单 bench 端到端验证**:选 gsm8k,用新结构起一次 run,对拍旧结果(能到 baseline/相近分即算通)。
- [x] **P6 — 去旧耦合**:删 `DataOS/resources/code/...` 与 `custom_operators`;H100 部署改为 checkout runtime 库(+ operator/task wheel 手搬流程固化);更新 `docs/ARCHITECTURE.md`。两库各自 commit。

**每个 P 结束都 commit,不一次性大改;有问题随时 `git reset --hard v0.3.0-opus-baseline`。**

---

## 5. 风险 / 注意

- **P3(拆 PLAYBOOK)最容易翻车**:环境和任务在原 PLAYBOOK 里缝在一起,拆点要清晰;拆完 sub-agent 读两份要能拼出和原来等价的指令。P5 端到端验证就是防这个。
- **不改运行中的东西**:现在没有 run 在跑(三个已结束),放心改;但改完 P5 要真起一次验证。
- **两库版本协调**靠 run-config `runtime_ref`,别退回"碰巧同源"。
