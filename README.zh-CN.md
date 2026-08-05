# DataOS — 自主 PostTrainBench Agent

[English README](README.md)

DataOS 在固定数据池和固定 GPU 时间预算内，把**一个** base model 在**一个** PostTrainBench
benchmark 上后训练到最优——全程由 agent 驱动。

一个常驻的 **leader**（code-agent CLI 对话——kimi / claude / codex / qwen，可按 run 配置）
维护一棵实验**树**。每一轮它提出一个实验——*数据配方*（用
[dataos-operator](tools/operator) 的算子在固定池上组 pipeline）加*训练*超参——由
**experimenter**（另一个 code-agent CLI，带 shell）端到端实现：

```
构建 SFT 数据（算子） → 训练（dataos-train / LLaMA-Factory，H100 Slurm）
        → 评测（冻结的 lm-eval PostTrainBench 协议） → 写 report.json
```

leader 复盘报告，树从最有希望的节点继续生长，直到预算耗尽。全树最优模型即提交产物。
第 0 轮之前还有一次性的 **investigator** 侦查（池画像 + 评测格式契约）。

## 架构

```
            ┌──────────────────────── agent 主机 ────────────────────────┐
  第 N 轮:  │  leader（CLI 对话） ──plan──▶ 代码：校验 · 加节点 · 注入父模型 │
            │        ▲                                            │      │
            │     review ◀── report.json ◀───────────────┐       ▼      │
            │                                            │   experimenter（CLI + shell） │
            └────────────────────────────────────────────┼──────────│──────────────────┘
                                                          │    ssh / slurm
                                                          │        ▼
                                                ┌─────────┴───────  H100 集群  ───────┐
                                                │ 数据算子 · 训练（冻结 venv） · vLLM 评测 │
                                                └─────────────────────────────────────────────┘
```

- **agent 主机**本地跑所有 agent。代码（`dataos.agent.loop.run_auto`）拥有实验树、计划
  校验、父模型注入、预算记账和持久化；LLM 只提计划和写复盘。
- **H100 集群**在冻结 uv venv 里做全部 GPU 工作（见
  `configs/environments/PTB_H100_slurm/`）；experimenter 经 SSH + Slurm 投递作业。
  judge 类 bench 在 GPU 节点生成、login 节点评分。
- **实时监控**（`src/dataos/agent/monitor.py` + `web/`）由 `run_auto` 内嵌：实验树、
  当前节点的 `data → train → eval` 子阶段进度条、各 agent 的推理+工具调用直播台，
  以及所有 agent transcript 静默时的停滞告警。一个 monitor 服务 workspace 下所有 run。
- 仓库以 git checkout 形式部署到 H100；**部署 = `git pull`**（工具版本变化时再跑
  `scripts/sync_envs.py`——见下文 `dataos_tools`）。

## 本仓库（monorepo）

一个仓、一个 uv workspace——没有任何外部 git pin：

- `src/dataos/`——大脑：循环、planner/reviewer、subagent 执行器、配置边界、prompt 模板
  （`src/dataos/prompt/{system,kb,task}/*.yaml`）、监控，以及
  `Project → Study → Experiment → Stage` 数据模型（`src/dataos/core/`）。
- `src/dataos_agent/`——code-agent CLI 驱动层（provider：claude / kimi / codex / qwen）：
  中立事件流、传输自愈（断连→退避→resume）、静默看门狗（`idle_timeout`→杀→续跑）。
  文档：`docs/dataos_agent/`。
- `tools/train/`——**dataos-train**：确定性训练启动器生命周期
  （validate / build / submit / status / collect；LLaMA-Factory + verl 后端）。
- `tools/operator/`——**dataos-operator**：数据配方算子目录
  （52 个算子：evaluator / mapper / mixer / reweighter / selector）。

集群上的工具版本由 env.yaml 的 `dataos_tools` 声明绑死（有测试保证与
`tools/*/pyproject.toml` 一致），用 `scripts/sync_envs.py` 同步。

## 前置条件

- [`uv`](https://docs.astral.sh/uv/)，Python ≥ 3.11。
- 一个在 agent 主机上已登录的 code-agent CLI（默认 `kimi`；也支持 `claude` / `codex` /
  `qwen`——用 run JSON 的 `agent` 块选择）。
- agent 主机到 GPU 集群（Slurm）的免密 SSH；冻结 venv、固定数据池、base model、
  benchmark 缓存都在集群上。

```bash
uv sync --extra dev     # 构建 agent 主机环境（只跑 agent；GPU 工作在 H100 上）
```

## 运行

一个 run 所需的全部是一个 JSON（`configs/runs/<run>.json`）——只有**事实**，没有策略：

```jsonc
{
  "goal": "用固定 PTB 数据池（v2）在 10h 预算内提升 Qwen3-1.7B-Base 的 gsm8k。",
  "workspace": "workspace/ptb-gsm8k",          // workspace 路径即 run 身份
  "environment": "PTB_H100_slurm",             // configs/environments/<env>/env.yaml
  "task": "posttrainbench",                    // configs/tasks/<task>/
  "bench": "gsm8k",                            // 唯一的 PostTrainBench 目标
  "inputs": {
    "base_model_path": "/data/.../Qwen3-1.7B-Base",
    "fixed_pool": "/data2/.../posttrainbench_fixed_pool_v2"
  },
  "exec": {
    "deploy_root": "/data/.../dataos_agentver",  // 集群上的仓库 checkout
    "runs_base": "/data2/.../runs",              // 集群上的实验产物区
    "resource": { "gpus": 1, "cpus": 8 }         // 可选 "node" 钉节点；不钉则 Slurm 自选
  },
  "budget": { "time_hours": 10, "eval_samples": -1 },   // -1 = 全量评测
  "agent": { "provider": "kimi", "model": null, "effort": null, "idle_timeout_s": 1800 },
  "monitor_port": 8765
}
```

```bash
uv run dataos configs/runs/ptb-gsm8k.json            # 全新实例（同名被占自动顺延 (1)(2)…）
uv run dataos configs/runs/ptb-gsm8k.json --resume   # 续跑同一 workspace
```

逐字段规范见 `configs/CONTRACT.md`（run JSON 见 §9）。所有**策略**——数据配方、训练方法
（LoRA vs 全量、超参）、从哪个节点生长——都是 leader 每轮的决策，持久化在 run
workspace 里。

## 一次 run 的流程

1. **investigator** 画像固定池、读取评测格式契约 → `recon.json`（每个 run 一次）。
2. 第 0 轮评测 **base model** 作为 baseline 参照。
3. 每轮代码把实验树 + 最近复盘发给 leader；它返回一个 `ExperimentPlan`（父节点 +
   数据配方 + 训练配置）。代码校验、加子节点、注入父模型路径。
4. **experimenter** 在 H100 上执行（data → train → eval → review），**增量式**写
   `report.json`——eval 的 `summary.json` 一出现分数立即落盘；若 CLI 中途死亡，循环
   会从盘上抢救已完成的分。
5. leader 复盘报告；代码记账并长树，直到**理想时间**预算耗尽（Slurm 排队时间有测量、
   不计入预算）。最优节点的模型写入 `final_model.json`。

健壮性内建：传输自愈重连断流的 CLI；静默看门狗（`agent.idle_timeout_s`，默认 1800s）
杀死并续跑静默的 CLI；监控对所有 agent 同时静默的 run 亮黄牌。

## 评测

冻结、自包含的 PostTrainBench 协议在 `configs/tasks/posttrainbench/eval/`（lm-eval
bridge），benchmark 数据随仓库 vendored 在 `configs/tasks/posttrainbench/benchmarks/`。
5 个确定性 bench（gsm8k、humaneval、bfcl、gpqamain、aime2025）在 GPU 作业内评分；2 个
judge bench（healthbench、arenahardwriting）在作业内生成、login 节点上用
OpenAI 兼容 judge 评分。协议决策与 baseline 快照见 `docs/eval/`。

## 文档地图

- `AGENTS.md`——AI coding agent 的操作指南（约定、不变量、H100 事实）
- `configs/CONTRACT.md`——env / task / run JSON 逐字段规范
- `docs/dataos_agent/`——CLI 驱动层（API、设计、用法）
- `tools/train/README.md` + `tools/train/docs/`——dataos-train（TrainSpec、online ops、坑）
- `tools/operator/README.md` + `tools/operator/docs/`——算子目录
- `docs/eval/`——评测协议决策 + baseline 结果
- `docs/troubleshooting.md`——真实运维坑（ssh、HF 下载、集群守护）
- `docs/known-issues.md`——run 复盘与修复记录

## 验证

```bash
uv run ruff check src tests
uv run pytest -q
```

## 许可证

MIT——见 `LICENSE`。
