# DataOS — Autonomous PostTrainBench Agent

[中文说明 / Chinese README](README.zh-CN.md)

DataOS post-trains **one** base model to maximize **one** PostTrainBench benchmark, within a
fixed data pool and a fixed GPU-time budget — driven end to end by agents.

A persistent **main-agent** (a Claude Code dialog) maintains an experiment **tree**. Each round
it proposes a single experiment — a *data recipe* (a [DataOS-Operator](https://github.com/USTC-StarTeam/DataOS-Operator)
pipeline over the fixed pool) plus *training* hyperparameters — and a **sub-agent** realizes it
end to end:

```
build SFT data (operators)  →  train with LLaMA-Factory on H100 (Slurm)
        →  evaluate with the lm-eval PostTrainBench protocol  →  write report.json
```

The main-agent then reviews the result and the tree grows from the most promising node, until
the wall-clock budget is spent. The best model over the tree is the submission.

## Architecture

```
            ┌──────────────────────── agent host (this machine) ────────────────────────┐
  round N:  │  main-agent (CC dialog) ──plan──▶ code: validate · add node · inject parent │
            │        ▲                                            │                       │
            │     review ◀── report.json ◀───────────────┐       ▼                       │
            │                                             │   sub-agent (CC + shell)      │
            └─────────────────────────────────────────────┼───────────│──────────────────┘
                                                           │     ssh / slurm / rsync
                                                           │           ▼
                                                 ┌─────────┴────────  H100 cluster  ───────┐
                                                 │ data ops · LLaMA-Factory train · vLLM eval │
                                                 └────────────────────────────────────────────┘
```

- The **agent host** runs *both* agents locally. Code (`dataos.agent.loop`) owns the tree, plan
  validation, parent-model injection, and persistence; the LLM only proposes plans and reviews.
- The **H100 cluster** does all GPU work. The sub-agent ships training (Slurm + LLaMA-Factory) and
  evaluation (vLLM + lm-eval) to H100 over SSH; judge benches score on the login node.
- The repo is deployed to H100 as a git checkout (`$AGENTVER`); **deploy = `git pull`**.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python env), Python ≥ 3.11.
- Claude Code CLI, authenticated — it is the main-agent and sub-agent backend.
- Passwordless SSH from the agent host to the H100 cluster (the cluster holds the prebuilt conda
  envs, the fixed data pool, the base models, and the benchmark cache).

```bash
uv sync          # build the agent-host env (the agent only; training/eval run in H100 conda envs)
```

## Run

Everything one run needs is a single JSON (`configs/runs/<run>.json`) — only **facts**, never strategy:

```jsonc
{
  "run_id": "ptb-h100-qwen-gsm8k",
  "workspace": "workspace/ptb-h100-qwen-gsm8k",
  "goal": "Use the fixed PTB data pool to improve the base model within a limited H100 budget.",
  "bench": "gsm8k",                       // the single PostTrainBench target
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

The config carries only facts (paths, the single bench, budget, resources). All **strategy** — the
data recipe, training method (LoRA vs full, hyperparameters), which tree node to grow, eval
sampling — is decided by the main-agent each round and persisted in the per-round plans under the
run workspace.

## How a run works

1. `freeze_eval` fixes the eval protocol once (the single bench + sampled limit).
2. Each round, code sends the main-agent the tree, recent reviews, and the operator catalog; it
   returns one `ExperimentPlan` (parent node + data recipe + train config). Code validates it, adds
   a child node, and injects the parent's model path (children continue from the parent model).
3. The sub-agent executes the plan on H100 following
   [`resources/code/posttrainbench/PLAYBOOK.md`](resources/code/posttrainbench/PLAYBOOK.md) and writes
   `report.json`.
4. The main-agent reviews the report; code attaches the score, grows the tree, and loops until the
   budget is spent. The best node's model is written to `final_model.json`.

## Evaluation

Self-contained **lm-evaluation-harness** PostTrainBench protocol under
[`resources/code/posttrainbench/lm_eval_bridge`](resources/code/posttrainbench/lm_eval_bridge):
5 deterministic benches (gsm8k, humaneval, bfcl, gpqamain, aime2025) scored in the GPU job, and
2 judge benches (healthbench, arenahardwriting) generated in the job then scored by an
OpenAI-compatible judge on the login node. Datasets resolve offline from the vendored benchmark
cache (`paths.benchmarks`).

## Repository layout

```text
src/dataos/
  agent/         the autonomous loop — loop.py (run_auto) · planner · reviewer · state_tree ·
                 run_config (the JSON config boundary) · prompts · backend · budget · logging
  core/          the Project → Study → Experiment → Stage / Data·Model·Eval abstraction + workspace/io
  runtime/       subagent_executor (per-experiment sub-agent driver) · eval_freeze
  integrations/  llm/ (Claude Code backend) · posttrainbench/ (eval protocol + bench list)
configs/
  runs/          one JSON per run (the user-facing config)
  experiments/   plan template
resources/
  code/posttrainbench/  PLAYBOOK.md (the sub-agent manual) · lm_eval_bridge/ (self-contained lm-eval eval)
  data/                 vendored datasets + posttrainbench_benchmarks (HF cache + judge data)
docs/            architecture notes (some predate the agent-version refactor)
scripts/         dev helpers
tests/           grouped by agent / core / runtime / integrations
```

## Deploy (H100)

The H100 copy at `$AGENTVER` is a git checkout of `origin/main`:

```bash
git push origin main                                   # from the agent host
ssh <h100> 'git -C /data/.../dataos_agentver pull'     # on the cluster
```

## Validation

```bash
uv run ruff check src tests
uv run pytest -q
```
