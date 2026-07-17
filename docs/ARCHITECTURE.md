# DataOS 总体设计架构

> 自主 SFT(post-training)研究运行时。给定一个基准(benchmark)和一个基座模型,DataOS 自动地"规划实验 → 造数据 → 训练 → 评测 → 复盘 → 迭代",在给定时间预算内自主搜索出能提升该基准分数的训练配方。
>
> 本文档描述**当前**架构(2026-07,以工具实际核实为准),不含历史遗留分支。

---

## 1. 一句话定位

**"Claude Code 当运行时,数据管线在算子里"**:核心不是一段写死的 `run_task()` 流程,而是一个由 LLM agent 驱动的自主循环——main-agent 负责"想",sub-agent 负责"做",人类只给一个基准 + 预算,其余全自动。

---

## 2. 两平面拓扑(控制面 / 执行面分离)

DataOS 刻意把"决策"和"算力"分开:

```
┌──────────────────────── 控制面 (5090, 无 GPU) ─────────────────────────┐
│  tangjiahui@server.starteam.wang:2213   repo: /zssd/.../2026/DataOS      │
│                                                                          │
│   dataos <run-config.json>  ─►  run_auto() 自主循环 (loop.py)            │
│        │                                                                 │
│        ├── main-agent  (规划 ExperimentPlan / 复盘 ReviewOut)            │
│        └── sub-agent   (执行:造数据→训练→评测→写 report.json)          │
│                 │  以上均为 claude_code 后端 → 本地 claude CLI 登录       │
│                 │  (Claude Opus 4.8 / effort=xhigh, 真 Anthropic 账号)   │
└─────────────────┼────────────────────────────────────────────────────────┘
                  │ ssh + sbatch (sub-agent 亲自驱动)
                  ▼
┌──────────────────────── 执行面 (H100 Slurm 集群) ──────────────────────┐
│  hcgu@114.214.215.159   node gpu04 (一个节点=多张 H100 卡)              │
│                                                                          │
│   sbatch GPU 作业内:  build data (dataos_operator) → SFT (LLaMA-Factory)│
│                        → sampled eval (lm-eval / PostTrainBench)         │
│   train_env / eval_env = /data/hcgu/.conda/envs/*                        │
│   dataos 核心经 PYTHONPATH=deploy_root/src 注入;operator 为 pip 包       │
│   fixed_pool (固定数据池) + base model 均在 H100 本地磁盘                 │
└──────────────────────────────────────────────────────────────────────────┘
```

- **控制面**:只跑 agent 大脑 + 编排循环,几乎不吃 GPU。轻量,可长时间挂机。
- **执行面**:所有训练/评测/数据处理都在 Slurm GPU 作业里跑,绝不在 login 节点上做重活。
- 二者唯一的耦合是 sub-agent 通过 SSH 把一个个 sbatch 作业推到 H100 并轮询到终态。

---

## 3. Agent 层(`src/dataos/agent/`)

后端只有一种:**`claude_code`** —— main-agent 和 sub-agent 都直接驱动官方 `claude-agent-sdk`(封装在 `ClaudeCodeAgent`),用本机 `claude` CLI 的登录认证。封装内收在 `src/dataos/agent/llm/`,不依赖任何外部 agent 包。

### 3.1 main-agent(大脑:规划 + 复盘)
- **持久 dialog**,只产出**结构化输出**(不直接碰文件):
  - `ExperimentPlan`:下一步做什么实验(选哪个源、什么训练配方、超参、假设)。
  - `ReviewOut`:一个实验跑完后的解读 + 下一步方向。
- 负责在**实验树**上做搜索决策(从哪个节点长、改哪个旋钮)。

### 3.2 sub-agent(执行体:把计划落地)
- 拿到一个 plan,**端到端**在 H100 上实现它:读真实数据行 → 用算子(+ 必要时手写脚本)造 SFT 数据 → sbatch 训练 → 采样评测 → 写 `report.json`。
- 有 shell、能 ssh/sbatch、能自愈(操作端口不匹配、格式不对等做最小兼容修复)。
- 提示词硬约束(`prompt/subagent_task.yaml`):算子箱缺什么就**手写**;失败要**诊断根因**再重试而非盲试;守 login-node 纪律;保持 GPU 利用率;无论如何都要写 report.json。

### 3.3 llm/ 支撑层
| 模块 | 职责 |
|---|---|
| `claude_code.py` | `ClaudeCodeAgent`:唯一后端类,直调 SDK `query()`;内含**传输层自愈**(断连→退避→`resume` 同一 session,main/sub 共享的基础能力) |
| `structured.py` | 结构化调用两层护栏:JSON 解析失败→纠正重问;传输错已下沉到 run() |
| `config.py` | `build_agent()`:main/sub 唯一构造入口,model 恒取 `cfg.claude_code_model` |
| `_output.py` | 自实现 `parse_output` + `OutputParseError`(从自由文本抽 JSON 校验进 Pydantic) |
| `net_retry.py` | 传输可恢复性白名单 `is_recoverable` + backoff;`ClaudeCodeError`(真错误,永不重试) |

---

## 4. 自主循环(`loop.py` `run_auto`)

```
RECON (强制一次)
   └─ recon sub-agent:读评测器代码 + 扫数据池 → recon.json
        · eval_format_contract  (打分器怎么解析、目标格式、format_sensitive、target_example)
        · baseline_diagnosis    (低分是格式问题还是能力问题)
        · source_profiles       (每个源的 quality / domain / target_ratio 画像)
   ▼
round N 循环 (直到 wall-clock 预算耗尽):
   main-agent 规划 plan ──► _validate_plan 校验 ──► sub-agent 执行 ──► 打分
        ▲                                                              │
        └────────────── main-agent 复盘 + 在实验树上长出新节点 ◄───────┘
```

- **实验树**:每个节点 = 一个实验(有父节点、有分数),从当前最好节点"生长"新实验;全程跟踪 `tree best`。
- **`_validate_plan` 校验闸门**:结构性检查(名字、超参)+ 源必须在 recon 画像里 + domain 必须相关。
  - > 注:原本还有一道"**DIRTY-format 源拒绝**"(target_ratio 太低就拒),2026-07-14 按需求**已移除**——格式不对的源也允许训练(可能学到别的信号),且避免"计划反复被拒→崩溃"。RECON 仍照常画像,但只作为给 main-agent 的**参考信息**,不再硬拦。
- 预算是硬上限(`wall_budget`,如 10h);检查发生在**轮次之间**。

---

## 5. 数据与算子

- **固定数据池 (fixed pool)**:PostTrainBench 预置的多来源原始数据,在 H100 本地。
- **`dataos_operator` (v0.5.0)**:数据算子工具箱——offline(map/dedup/质量评分 evaluator/选择 selector/多样性/混合)+ online(训练期 loss/doremi 等 reweight/select)。sub-agent 把它们**按名字组成一条 recipe** 造 SFT 数据。
- **格式对齐**是关键一环:把训练目标渲染成**评测器能解析的精确形状**(如 BFCL 的 `<tool_call>` + qwen3 模板)。算子箱没有的清洗/对齐逻辑,sub-agent 手写小脚本补(可沉淀为 run-local 或共享算子)。
- 最终统一 bridge 成 **LLaMA-Factory alpaca 格式**喂训练。

---

## 6. 评测(PostTrainBench)

- 每个 bench 有**冻结的评测协议 + 打分器**,base 模型和 SFT 后模型走同一套。
- 两类打分:
  - **确定性规则**(gsm8k / bfcl):抽答案/解析 tool-call 比对,本地秒判。
  - **LLM judge**(healthbench / arenahardwriting):每条回答按 rubric 多准则逐条让 judge 模型打分——慢(上千次 API 调用,样本级 32 并发、单样本内准则串行)。
- 评测在 sbatch GPU 作业内做生成(vLLM),judge 走 gateway API。

---

## 7. 配置与环境

- **一个 run-config JSON**(`configs/runs/ptb-h100-qwen-<bench>.json`)→ `run_config.py` → `AutoRunConfig`。含:base_model_path、bench、resource、runtime(cluster / gateway / judge / paths)。
- **`~/.dataos_deepseek.env`**(不入 git):`DATAOS_BACKEND=claude_code`、`DATAOS_OPERATOR_PLUGINS=…`。
  - 曾用来把 claude 重定向到 DeepSeek `/anthropic` 的 `ANTHROPIC_*` 覆盖**已注释**——现在直接用本机 claude CLI 登录(真 Anthropic)。
- **大脑模型**:`config.py` 的 `claude_code_model`(现为 `claude-opus-4-8`)+ `~/.claude/settings.json` 的 `effortLevel=xhigh` → main/sub 全部 **Opus 4.8 / xhigh**。

---

## 8. 关键设计取舍

| 取舍 | 理由 |
|---|---|
| 控制面/执行面分离 | 大脑轻量长挂;算力集中在 Slurm,守 login-node 纪律 |
| 单一 `claude_code` 后端,内收封装 | 核心逻辑一个仓库,只依赖官方 SDK,不分散 |
| main "想" / sub "做" 二分 | 规划与执行解耦;sub 有 shell 能自愈,plan 只描述意图 |
| RECON 前置 + 源画像 | 让 agent 有据可依地选源/对齐格式,避免"在脏数据上瞎训" |
| 传输自愈下沉进 run() | 断连重连是 agent 层基础能力,main/sub 自动获得 |
| 预算为硬上限、树式搜索 | 有限算力内自主探索,从最好节点生长 |
