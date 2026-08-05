# DataOS — Autonomous PostTrainBench Agent

[中文说明 / Chinese README](README.zh-CN.md)

DataOS post-trains **one** base model to maximize **one** PostTrainBench benchmark, within a
fixed data pool and a fixed GPU-time budget — driven end to end by agents.

A persistent **leader** (a code-agent CLI dialog — kimi / claude / codex / qwen, configurable
per run) maintains an experiment **tree**. Each round it proposes a single experiment — a *data
recipe* (a pipeline of [dataos-operator](tools/operator) operators over the fixed pool) plus
*training* hyperparameters — and an **experimenter** (another code-agent CLI, with a shell)
realizes it end to end:

```
build SFT data (operators)  →  train (dataos-train / LLaMA-Factory, Slurm on H100)
        →  evaluate (frozen lm-eval PostTrainBench protocol)  →  write report.json
```

The leader reviews the report and the tree grows from the most promising node, until the
budget is spent. The best model over the tree is the submission. A one-shot **investigator**
gathers pre-training intel (pool profiles + the eval's format contract) before round 0.

## Architecture

```
            ┌──────────────────────── agent host ────────────────────────┐
  round N:  │  leader (CLI dialog) ──plan──▶ code: validate · add node · inject parent │
            │        ▲                                            │      │
            │     review ◀── report.json ◀───────────────┐       ▼      │
            │                                            │   experimenter (CLI + shell)  │
            └────────────────────────────────────────────┼──────────│──────────────────┘
                                                          │    ssh / slurm
                                                          │        ▼
                                                ┌─────────┴───────  H100 cluster  ───────┐
                                                │ data ops · train (frozen venvs) · vLLM eval │
                                                └─────────────────────────────────────────────┘
```

- The **agent host** runs all agents locally. Code (`dataos.agent.loop.run_auto`) owns the
  tree, plan validation, parent-model injection, budget accounting, and persistence; the LLM
  only proposes plans and writes reviews.
- The **H100 cluster** does all GPU work in frozen uv venvs (see
  `configs/environments/PTB_H100_slurm/`); the experimenter ships jobs over SSH + Slurm.
  Judge benches generate on the GPU node and score on the login node.
- A **live monitor** (`src/dataos/agent/monitor.py` + `web/`) is embedded by `run_auto` on one
  port per run: experiment tree, the running node's `data → train → eval` sub-stage stepper,
  a live console of each agent's reasoning + tool calls, and a stall alert when every agent
  transcript goes silent. One monitor serves every run under the workspace.
- The repo is deployed to H100 as a git checkout; **deploy = `git pull`** (plus
  `scripts/sync_envs.py` when a tool version changed — see `dataos_tools` below).

## This repository (monorepo)

One repo, one uv workspace — no external git pins:

- `src/dataos/` — the brain: the loop, planner/reviewer, subagent executor, config boundary,
  prompt templates (`src/dataos/prompt/{system,kb,task}/*.yaml`), the monitor, and the
  `Project → Study → Experiment → Stage` data model (`src/dataos/core/`).
- `src/dataos_agent/` — the code-agent CLI driver layer (providers: claude / kimi / codex /
  qwen): neutral event stream, transport self-healing (endpoint drop → backoff → resume),
  and a stall watchdog (`idle_timeout` → kill → resume). Docs: `docs/dataos_agent/`.
- `tools/train/` — **dataos-train**: the deterministic training launcher lifecycle
  (validate / build / submit / status / collect; LLaMA-Factory + verl backends).
- `tools/operator/` — **dataos-operator**: the data-recipe operator catalog
  (52 operators: evaluators, mappers, mixers, reweighters, selectors).

Tool versions on the cluster are pinned by the `dataos_tools` declaration in `env.yaml`
(kept equal to `tools/*/pyproject.toml` by a test) and synced with `scripts/sync_envs.py`.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/), Python ≥ 3.11.
- A code-agent CLI, authenticated on the agent host (`kimi` by default; `claude` / `codex` /
  `qwen` also supported — pick per run via the `agent` block).
- Passwordless SSH from the agent host to the GPU cluster (Slurm), where the frozen venvs,
  the fixed data pool, the base models, and the benchmark cache live.

```bash
uv sync --extra dev     # build the agent-host env (agents only; GPU work runs on H100)
```

## Run

Everything one run needs is a single JSON (`configs/runs/<run>.json`) — only **facts**, never
strategy:

```jsonc
{
  "goal": "Use the fixed PTB data pool (v2) to improve Qwen3-1.7B-Base on gsm8k within 10h.",
  "workspace": "workspace/ptb-gsm8k",          // the workspace path IS the run identity
  "environment": "PTB_H100_slurm",             // configs/environments/<env>/env.yaml
  "task": "posttrainbench",                    // configs/tasks/<task>/
  "bench": "gsm8k",                            // the single PostTrainBench target
  "inputs": {
    "base_model_path": "/data/.../Qwen3-1.7B-Base",
    "fixed_pool": "/data2/.../posttrainbench_fixed_pool_v2"
  },
  "exec": {
    "deploy_root": "/data/.../dataos_agentver",  // the repo checkout on the cluster
    "runs_base": "/data2/.../runs",              // per-experiment scratch on the cluster
    "resource": { "gpus": 1, "cpus": 8 }         // optional "node" pin; else Slurm picks
  },
  "budget": { "time_hours": 10, "eval_samples": -1 },   // -1 = full benchmark
  "agent": { "provider": "kimi", "model": null, "effort": null, "idle_timeout_s": 1800 },
  "monitor_port": 8765
}
```

```bash
uv run dataos configs/runs/ptb-gsm8k.json            # fresh instance (occupied names get (1), (2)…)
uv run dataos configs/runs/ptb-gsm8k.json --resume   # continue the same workspace
```

The field-by-field spec is `configs/CONTRACT.md` (§9 for the run JSON). All **strategy** — the
data recipe, training method (LoRA vs full, hyperparameters), which tree node to grow — is the
leader's per-round decision, persisted in the run workspace.

## How a run works

1. The **investigator** profiles the fixed pool and reads the eval's format contract →
   `recon.json` (once per run).
2. Round 0 evaluates the **base model** as the baseline reference.
3. Each round, code sends the leader the tree + recent reviews; it returns one
   `ExperimentPlan` (parent node + data recipe + train config). Code validates it, adds a
   child node, and injects the parent's model path.
4. The **experimenter** executes the plan on H100 (data → train → eval → review), writing
   `report.json` **incrementally** — the score hits disk the moment the eval's `summary.json`
   exists; if the CLI dies mid-run, the loop salvages the completed score from disk.
5. The leader reviews the report; code attaches the score and grows the tree, until the
   **ideal-time** budget is spent (Slurm queue wait is measured and excluded). The best
   node's model is written to `final_model.json`.

Robustness is built in: transport self-healing reconnects dropped CLI streams; a stall
watchdog (`agent.idle_timeout_s`, default 1800s) kills and resumes a silent CLI; the monitor
flags runs whose agents have all gone quiet.

## Evaluation

The frozen, self-contained PostTrainBench protocol lives in
`configs/tasks/posttrainbench/eval/` (lm-eval bridge), with the benchmark data vendored at
`configs/tasks/posttrainbench/benchmarks/` (deployed with the repo). 5 deterministic benches
(gsm8k, humaneval, bfcl, gpqamain, aime2025) score inside the GPU job; 2 judge benches
(healthbench, arenahardwriting) generate in the job then score by an OpenAI-compatible judge
on the login node. Protocol decisions and baseline snapshots: `docs/eval/`.

## Documentation map

- `AGENTS.md` — the operating guide for AI coding agents (conventions, invariants, H100 facts)
- `configs/CONTRACT.md` — env / task / run-JSON field-by-field spec
- `docs/dataos_agent/` — the CLI driver layer (API, design, usage)
- `tools/train/README.md` + `tools/train/docs/` — dataos-train (TrainSpec, online ops, pitfalls)
- `tools/operator/README.md` + `tools/operator/docs/` — the operator catalog
- `docs/eval/` — the eval protocol decisions + baseline results
- `docs/troubleshooting.md` — real operational gotchas (ssh, HF downloads, cluster guards)
- `docs/known-issues.md` — run postmortems and their fixes

## Validation

```bash
uv run ruff check src tests
uv run pytest -q
```

## License

MIT — see `LICENSE`.
