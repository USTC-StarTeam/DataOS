# DataOS

**面向大模型后训练的自主研究循环。** 给它一个基座模型、一个固定数据池、一个基准和一份 GPU 时间预算，
它自己设计实验、构建数据、训练、按冻结协议评估、复盘自己的结果，并把实验树生长到预算耗尽——
最后交回最优的那个检查点。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)

[English README](README.md)

```bash
uv sync --group dev
uv run dataos configs/runs/ptb-gsm8k.json
```

---

## 它做什么

一次运行只针对**一个**基座模型和**一个**基准。每一轮系统提出一个实验——固定池上的一条数据配方，
加上训练超参——端到端地实现它、打分，然后决定下一步往哪里生长：

```
构建 SFT 数据（算子）→ 训练（LLaMA-Factory / verl，Slurm）→ 评估（冻结的 lm-eval）→ report.json → 复盘
```

搜索过程没有任何脚本化的部分。混哪些数据源、用 LoRA 还是全参、一次回退之后该试什么、什么时候放弃
一条死掉的分支——全部由每一轮根据磁盘上的证据现场决定。

## 设计上的几条承诺

下面这几条决定了其余一切。每一条的存在，都是因为它缺席时我们付出过可计量的算力代价。

**模型提议，代码决策。** LLM 每轮返回一份 `ExperimentPlan`。实验树、计划校验、父模型注入、预算记账
与持久化都归 Python 所有。格式错误或非法的计划会被拒绝并重新询问，绝不半途应用。状态落在磁盘上，
所以被杀掉的运行是从它停下的地方恢复，而不是从某段对话记得的内容恢复。

**评估协议是冻结的，且由配置而不是 agent 决定。** 对话模板、题量、并发、judge 路由和 runner 后端
都声明在 `configs/tasks/<task>/eval/protocol.yaml` 里，由评估器自己读取。对话模板从模型自己的
`config.json` 解析，绝不从调用方传入的标签推断——**一个 agent 能选的标签，就是一个 agent 能挪动的协议**。
分数在任何展示处都带着它的口径（`74.00@200/1319`），所以子集分永远不会被当成全量分读。

**每个实验只产出一份带类型的报告，下一轮完整地读它。** `report.json` 按产生它的四个阶段
（data / train / eval / review）成形，每个阶段各自带着自己的状态、它需要的绕行处理，以及它测到的
数字——实际完成的 epoch 数、最终训练 loss、训练真正使用的对话模板、指标的标准误。中间没有手挑的
摘要层，因为投影的键名会悄无声息地与契约错开。

**三个平级角色，一个编排者。** **leader** 负责规划与复盘；**experimenter** 端到端实现一个节点；
**investigator** 在第 0 轮之前画像数据池、读懂评分器的解析逻辑。三者谁也不能发起谁——
它们都由 Python 循环调用。

## 架构

```
┌─────────────────────────── agent 主机 ───────────────────────────┐
│                                                                  │
│   leader ──── 计划 ────▶ loop：校验 · 生长树 · 记账                │
│     ▲                              │                             │
│     └──── 复盘 ◀── report.json ────┤                             │
│                                    ▼                             │
│                              experimenter ─────┐                 │
│                              investigator ─────┤                 │
└────────────────────────────────────────────────┼─────────────────┘
                                                 │ ssh + Slurm
                                                 ▼
┌──────────────────────────── GPU 集群 ────────────────────────────┐
│    数据算子   ·   训练（冻结 venv）   ·   vLLM 评估               │
└──────────────────────────────────────────────────────────────────┘
```

agent 主机运行所有 agent 并持有全部持久状态。集群在 `configs/environments/<env>/env.yaml` 声明的
冻结环境里完成每一个吃 GPU 的步骤；仓库以 git checkout 的形式部署在那里，所以部署就是 `git pull`
（工具版本变化时再加一次 `scripts/sync_envs.py`）。

循环自带**实时监控**（`src/dataos/agent/monitor.py` + `web/`）：实验树、当前节点的阶段进度、
每个 agent 的推理与工具调用控制台，以及所有转录本同时静默时的停滞告警。一个监控服务于同一工作区下的
所有运行。

## 环境要求

- agent 主机上的 [`uv`](https://docs.astral.sh/uv/) 与 Python ≥ 3.11。
- 一个已认证的 code-agent CLI——`claude`、`kimi`、`codex` 或 `qwen`，可按运行逐个选择。
- 到 Slurm 集群的免密 SSH，集群上存放冻结环境、数据池、基座模型和基准缓存。

agent 主机本身不需要 GPU：它只跑 agent 和记账，每个吃 GPU 的步骤都提交到集群。

## 配置一次运行

每次运行一份 JSON，**只写事实，绝不写策略**：

```jsonc
{
  "goal": "用固定池在 10 GPU 小时内提升 Qwen3-1.7B-Base 的 gsm8k 成绩。",
  "workspace": "workspace/ptb-gsm8k",   // 工作区路径就是这次运行的身份
  "environment": "PTB_H100_slurm",      // configs/environments/<env>/env.yaml
  "task": "posttrainbench",             // configs/tasks/<task>/
  "bench": "gsm8k",                     // 唯一的目标基准
  "inputs": {
    "base_model_path": "/data/.../Qwen3-1.7B-Base",
    "fixed_pool": "/data2/.../posttrainbench_fixed_pool_v2"
  },
  "exec": {
    "deploy_root": "/data/.../dataos",  // 集群上的仓库 checkout
    "runs_base": "/data2/.../runs",     // 集群上的每实验工作目录
    "resource": { "gpus": 1, "cpus": 8 }
  },
  "budget": { "time_hours": 10, "eval_samples": -1 },   // -1 = 全量基准
  "agent": { "provider": "claude", "idle_timeout_s": 1800 },
  "monitor_port": 8765
}
```

```bash
uv run dataos configs/runs/ptb-gsm8k.json            # 全新实例
uv run dataos configs/runs/ptb-gsm8k.json --resume   # 续跑已有工作区
```

每个字段的规格见 [`configs/CONTRACT.md`](configs/CONTRACT.md)（§9 是 run JSON）。
**策略从不进配置**——它是 leader 每一轮的决定，持久化在工作区里。

## 一次运行是怎么走的

1. **investigator** 画像固定池、读懂评分器的解析逻辑，然后评估基座模型建立基线。两者都在任何训练
   开始之前落进工作区。
2. 每一轮，循环把实验树和最近的报告交给 leader，它返回一份 `ExperimentPlan`。循环校验它、添加子节点、
   注入父节点的模型路径。
3. **experimenter** 在集群上实现这份计划，并**增量写** `report.json`——分数在评估器的 `summary.json`
   出现的那一刻就落盘，所以一个死在分析中途的 CLI 丢掉的是复盘，绝不是测量结果。
4. leader 复盘；循环挂上分数并生长实验树。Slurm 排队时间会被测量并从预算中扣除，所以资源竞争消耗的是
   墙钟时间，而不是实验次数。
5. 预算耗尽时，最优节点的模型记录进 `final_model.json`。

失效处理是循环的一部分而非事后补丁：断掉的 CLI 流会重连并恢复同一会话；静默超过 `idle_timeout_s`
的 agent 会被杀掉并恢复；评估已完成但报告未终结的运行，其分数会从磁盘上被打捞回来。

## 仓库结构

单一 [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)，无外部 git 引用。

| 路径 | 内容 |
|---|---|
| `src/dataos/` | 循环、规划器与复盘器、角色执行器、冻结评估边界、`Project → Study → Experiment → Stage` 模型、报告 schema、提示词与监控 |
| `src/dataos_agent/` | code-agent CLI 驱动层：四个 provider 之上的中立事件流、传输自愈、停滞看门狗 |
| `tools/src/dataos_train/` | **dataos-train** —— 确定性的训练生命周期（validate / build / submit / status / collect），支持 LLaMA-Factory 与 verl |
| `src/operator/` | **dataos-operator** —— 52 个数据配方算子（27 selector、10 evaluator、10 mapper、4 mixer、1 reweighter） |
| `tools/src/dataos_eval/` | **dataos-eval** —— 提交并收集一次冻结基准评估，协议从配置加模型自己的 `config.json` 解析 |
| `configs/` | 环境、任务（基准、评估协议、随仓库分发的基准数据）与运行定义 |

提示词按角色与用途分开：`src/dataos/prompt/<role>/context/` 是这个角色始终知道的东西，
`<role>/turn/` 是这一轮要它做的事。集群上的工具版本由 `env.yaml` 里的 `dataos_tools` 声明钉住
（有测试保证它与各 `pyproject.toml` 相等），用 `scripts/sync_envs.py` 应用。

## 评估协议

PostTrainBench 的实现自包含在 `configs/tasks/posttrainbench/eval/`，基准数据随仓库放在
`configs/tasks/posttrainbench/benchmarks/`，所以一次运行不需要联网。五个基准（GSM8K、HumanEval、
BFCL、GPQA-main、AIME 2025）在 GPU 作业内部打分；两个 judge 基准（HealthBench、ArenaHard-writing）
在作业里只生成，之后由 OpenAI 兼容的 judge 打分。协议决策及其依据记录在 `docs/eval/`。

## 开发

```bash
uv run ruff check src tests
uv run pytest -q
```

测试套件很快（不需要 GPU、不联网），并且是各层之间的契约：它把报告 schema 钉在索要它的提示词上、
把训练 spec 钉在 `dataos-train` 自己的校验器上、把声明的工具版本钉在各个包上、
把评估协议钉在评估器解析出的值上。

`AGENTS.md` 是给在本仓库工作的 AI 编码 agent 的操作指南——约定、不变量，以及用代价换来的集群事实。
人类贡献者读它同样有用。

## 文档

| 文档 | 内容 |
|---|---|
| [`configs/CONTRACT.md`](configs/CONTRACT.md) | 环境 / 任务 / run JSON 的字段规格 |
| [`AGENTS.md`](AGENTS.md) | 在本仓库改代码的约定与不变量 |
| [`src/dataos-tools/docs/train/README.md`](src/dataos-tools/docs/train/README.md) | TrainSpec、后端、在线算子 |
| [`src/operator/README.md`](src/operator/README.md) | 算子目录 |
| `docs/eval/` | 评估协议决策与基线结果 |
| `docs/dataos_agent/` | CLI 驱动层 |
| `docs/troubleshooting.md` | 真实的运维失效模式及其修法 |

## 状态

Alpha（`0.3.0a0`）。循环、工具和 PostTrainBench 任务处于日常使用中；层与层之间的接口足够稳定到
可以在其上构建，但尚未冻结。欢迎 issue 与 PR——请先跑 `ruff` 与 `pytest`，
并说明这个改动意在**防止**什么，而不只是它做了什么。

## 许可

MIT —— 见 [`LICENSE`](LICENSE)。
