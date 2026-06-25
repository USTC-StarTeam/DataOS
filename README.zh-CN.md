# DataOS — 自主 PostTrainBench Agent

[English README](README.md)

DataOS 在**固定数据池**和**固定 GPU 时间预算**下，把**一个** base 模型 post-train 到在
**一个** PostTrainBench bench 上分数最高 —— 全程由 agent 驱动。

一个常驻的 **main-agent**（Claude Code 长对话）维护一棵实验**树**。每一轮它给出一个实验方案 ——
一份*数据配方*（对固定池跑一条 [DataOS-Operator](https://github.com/USTC-StarTeam/DataOS-Operator)
算子流水线）加一组*训练*超参 —— 由 **sub-agent** 端到端实现：

```
构建 SFT 数据(算子)  →  在 H100 上用 LLaMA-Factory 训练(Slurm)
        →  用 lm-eval 的 PostTrainBench 协议评测  →  写 report.json
```

main-agent 复盘结果，树从最有潜力的节点继续生长，直到耗尽时间预算。树上最优的模型即为提交。

## 架构

```
            ┌──────────────────────── agent host(本机) ────────────────────────┐
  第 N 轮:  │  main-agent(CC 对话) ──plan──▶ code: 校验 · 建节点 · 注入父模型     │
            │        ▲                                       │                  │
            │     复盘 ◀── report.json ◀───────────┐         ▼                  │
            │                                       │   sub-agent(CC + shell)   │
            └───────────────────────────────────────┼──────────│────────────────┘
                                                     │   ssh / slurm / rsync
                                                     │          ▼
                                           ┌─────────┴────────  H100 集群  ───────┐
                                           │ 数据算子 · LLaMA-Factory 训练 · vLLM 评测 │
                                           └────────────────────────────────────────┘
```

- **agent host** 在本地跑 *两个* agent。代码（`dataos.agent.loop`）负责树、plan 校验、父模型注入、
  落盘；LLM 只负责出 plan 和复盘。
- **H100 集群** 做全部 GPU 活。sub-agent 通过 SSH 把训练（Slurm + LLaMA-Factory）和评测
  （vLLM + lm-eval）丢到 H100；judge bench 在登录节点打分。
- 仓库以 git checkout 形式部署到 H100（`$AGENTVER`）；**部署 = `git pull`**。

## 前置条件

- [`uv`](https://docs.astral.sh/uv/)（Python 环境），Python ≥ 3.11。
- 已登录的 Claude Code CLI —— 它是 main-agent 和 sub-agent 的后端。
- 从 agent host 到 H100 集群的免密 SSH（集群上有预建的 conda 环境、固定数据池、base 模型、benchmark 缓存）。

```bash
uv sync          # 构建 agent host 环境(只装 agent；训练/评测跑在 H100 的 conda 环境里)
```

## 运行

一次运行所需的一切就是一个 JSON（`configs/runs/<run>.json`）—— 只放**事实**，不放策略：

```jsonc
{
  "run_id": "ptb-h100-qwen-gsm8k",
  "workspace": "workspace/ptb-h100-qwen-gsm8k",
  "goal": "在受限 H100 预算内用固定 PTB 数据池提升 base 模型。",
  "bench": "gsm8k",                       // 唯一的 PostTrainBench 目标
  "base_model_path": "/.../Qwen3-1.7B-Base",
  "time_budget_hours": 10,
  "resource": { "node": "gpu04", "gpus": 1, "cpus": 8 },
  "paths": {
    "fixed_pool":  "/data/.../posttrainbench_fixed_pool_v1_full",
    "conda_sh":    "/.../miniconda3/etc/profile.d/conda.sh",
    "train_env":   "/.../envs/Llama",     // LLaMA-Factory + torch
    "eval_env":    "/.../envs/lm-eval",   // lm-eval + vLLM
    "benchmarks":  "/data/.../posttrainbench_benchmarks"
  }
}
```

```bash
uv run dataos configs/runs/ptb-h100-qwen-gsm8k.json
```

config 只承载事实（路径、唯一 bench、预算、资源）。所有**策略** —— 数据配方、训练方式
（LoRA / 全参、超参）、选树上哪个节点生长、eval 采样 —— 都由 main-agent 每轮自主决定，并写进
运行 workspace 下的每轮 plan。

## 一次运行是怎么跑的

1. `freeze_eval` 一次性冻结评测协议（唯一 bench + 采样 limit）。
2. 每轮代码把树、最近的复盘、算子目录发给 main-agent；它返回一个 `ExperimentPlan`（父节点 +
   数据配方 + 训练配置）。代码校验、加子节点、注入父模型路径（子节点在父模型上续训）。
3. sub-agent 在 H100 上按
   [`resources/code/posttrainbench/PLAYBOOK.md`](resources/code/posttrainbench/PLAYBOOK.md) 执行计划，
   写出 `report.json`。
4. main-agent 复盘 report；代码附上分数、生长树，循环直到预算耗尽。最优节点的模型写入
   `final_model.json`。

## 评测

自包含的 **lm-evaluation-harness** PostTrainBench 协议，位于
[`resources/code/posttrainbench/lm_eval_bridge`](resources/code/posttrainbench/lm_eval_bridge)：
5 个确定性 bench（gsm8k、humaneval、bfcl、gpqamain、aime2025）在 GPU job 内打分，
2 个 judge bench（healthbench、arenahardwriting）在 job 内生成、在登录节点用 OpenAI 兼容 judge 打分。
数据集从 vendored benchmark 缓存（`paths.benchmarks`）离线解析。

## 仓库结构

```text
src/dataos/
  agent/         自主循环 —— loop.py (run_auto) · planner · reviewer · state_tree ·
                 run_config(JSON 配置边界) · prompts · backend · budget · logging
  core/          Project → Study → Experiment → Stage / Data·Model·Eval 抽象 + workspace/io
  runtime/       subagent_executor(逐实验 sub-agent 驱动) · eval_freeze
  integrations/  llm/(Claude Code 后端) · posttrainbench/(评测协议 + bench 列表)
configs/
  runs/          每次运行一个 JSON(面向用户的配置)
  experiments/   plan 模板
resources/
  code/posttrainbench/  PLAYBOOK.md(sub-agent 手册) · lm_eval_bridge/(自包含 lm-eval 评测)
  data/                 vendored 数据集 + posttrainbench_benchmarks(HF 缓存 + judge 数据)
docs/            架构说明(部分早于 agent-version 重构)
scripts/         开发辅助
tests/           按 agent / core / runtime / integrations 分组
```

## 部署(H100)

H100 上 `$AGENTVER` 是 `origin/main` 的 git checkout：

```bash
git push origin main                                   # 在 agent host
ssh <h100> 'git -C /data/.../dataos_agentver pull'     # 在集群上
```

## 校验

```bash
uv run ruff check src tests
uv run pytest -q
```
