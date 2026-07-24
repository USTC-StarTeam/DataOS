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
- A **live monitor** (`src/dataos/agent/monitor.py` + `web/`) is embedded by `run_auto` on one port
  (default `8765`): open `http://localhost:8765` to watch the experiment tree grow, the running
  node's `data → train → eval` sub-stage, and a live console of both agents' reasoning + tool calls.
  One monitor serves every run under the workspace.
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
    "eval_env":    "/.../envs/lm-eval"    // lm-eval + vLLM
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
3. The sub-agent executes the plan on H100. Its prompt carries two sections rendered from the
   DataOS-Runtime typed contract (`env.yaml` + `task.yaml`, via the runtime's `compose`):
   ENVIRONMENT OPS (filled submit/poll/fetch/activate commands + cluster discipline) and TASK
   (goal + frozen-eval scoring). The data-building/training METHOD and the report.json shape
   stay in the brain's own prompts. It writes `report.json`.
4. The main-agent reviews the report; code attaches the score, grows the tree, and loops until the
   budget is spent. The best node's model is written to `final_model.json`.

Watch it live at `http://localhost:8765` (set `monitor_port` in the config to change it, or
`DATAOS_MONITOR_PORT=0` to disable). Run several experiments under one workspace and the single
monitor shows them all, each marked live or done by its heartbeat.

## Evaluation

Self-contained **lm-evaluation-harness** PostTrainBench protocol, now living in the
agent-agnostic runtime repo (`DataOS-Runtime/tasks/posttrainbench/lm_eval_bridge`, read from
`runtime_root` on the exec host):
5 deterministic benches (gsm8k, humaneval, bfcl, gpqamain, aime2025) scored in the GPU job, and
2 judge benches (healthbench, arenahardwriting) generated in the job then scored by an
OpenAI-compatible judge on the login node. Datasets resolve offline from the repo's vendored
benchmark cache (`resources/data/posttrainbench_benchmarks`, deployed with the code). Eval requests
are batched into the vLLM server at `--num-concurrent` (default 32) for throughput — not
score-affecting.

## Repository layout

```text
DataOS/
├── src/dataos/
│   ├── cli.py                       # `dataos` entrypoint: load the run config → run_auto
│   ├── agent/                       # the autonomous loop + per-experiment sub-agent
│   │   ├── loop.py                  #   run_auto: each round = plan → execute → review → grow tree
│   │   ├── state_tree.py            #   the experiment tree (nodes, growable set, best node)
│   │   ├── run_config.py            #   the single-JSON config boundary (facts only)
│   │   ├── planner.py · reviewer.py #   ask the main-agent for an ExperimentPlan / ReviewOut
│   │   ├── subagent_executor.py     #   drives the per-experiment sub-agent (Claude Code + shell)
│   │   ├── eval_freeze.py           #   freeze the eval protocol once per run
│   │   ├── backend.py               #   builds the role agent (renders the role prompt)
│   │   ├── llm/                     #   code-agent CLI driver (cli_agent, via dataos-agent) + structured-output helper
│   │   ├── posttrainbench.py        #   PTB eval protocol (bench list + sampled limits per profile)
│   │   ├── schemas.py               #   ExperimentPlan / ReviewOut (validated JSON)
│   │   ├── monitor.py               #   live HTTP monitor (one port serves every run; heartbeats)
│   │   └── config.py · budget.py · logging.py
│   ├── prompt/                      # every prompt as a Jinja2 YAML template (rendered with config facts)
│   │   ├── main_agent.yaml · subagent.yaml            #   system prompts
│   │   ├── strategy.yaml                              #   strategy playbook (appended to main_agent)
│   │   ├── planner_task.yaml · reviewer_task.yaml     #   per-round task messages
│   │   ├── subagent_task.yaml · subagent_continue.yaml#   per-experiment task / resume
│   │   └── structured_*.yaml                          #   JSON-output scaffolding
│   └── core/                        # the Project → Study → Experiment → Stage data model
│       ├── project.py · study.py · experiment.py · stage.py
│       ├── artifact.py              #   Data · Model · Eval artifacts
│       ├── workspace.py             #   on-disk run layout
│       ├── catalog.py · loading.py  #   operator catalog + dataset loading (DataOS-Operator)
│       └── report.py · provenance.py · io.py · ids.py · redact.py · env.py
├── configs/
│   └── runs/ptb-h100-qwen-gsm8k.json #  the user-facing run config (one JSON per run)
├── resources/
│   ├── custom_operators/            #   DataOS-private sedimented operators (DATAOS_OPERATOR_PLUGINS)
│   └── data/
│       ├── posttrainbench_benchmarks/ # hf_datasets/ (offline HF cache) + local/ (judge data)
│       └── datasets/                  # small vendored datasets
│   # eval bridge + env/task contracts moved to the DataOS-Runtime repo (tasks/…/lm_eval_bridge)
├── web/                             # live monitor UI (index.html) + standalone server.py
├── docs/                            # architecture notes (some predate the agent-version refactor)
├── tests/                           # agent / core
├── scripts/                         # dev helpers (doctor.py)
├── pyproject.toml · uv.lock
└── README.md · README.zh-CN.md · AGENTS.md
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
