# DataOS

**An autonomous research loop for LLM post-training.** Give it a base model, a frozen data pool,
a benchmark and a GPU-hour budget. It designs experiments, builds the data, trains, evaluates
against a frozen protocol, reviews its own results, and grows a search tree until the budget is
spent — then hands back the best checkpoint.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)

[中文说明 / Chinese README](README.zh-CN.md)

```bash
uv sync --extra dev
uv run dataos configs/runs/ptb-gsm8k.json
```

---

## What it does

One run targets **one** base model and **one** benchmark. Each round the system proposes a single
experiment — a data recipe over the fixed pool, plus training hyperparameters — realizes it end to
end, scores it, and decides where to grow next:

```
build SFT data (operators) → train (LLaMA-Factory / verl, Slurm) → evaluate (frozen lm-eval) → report.json → review
```

Nothing about the search is scripted. Which sources to mix, LoRA or full fine-tuning, what to try
after a regression, when to stop a dead branch — all of it is decided per round from the evidence
on disk.

## Design commitments

These are the choices that shape everything else, and each exists because its absence cost us a
measurable amount of compute.

**The model proposes; code decides.** An LLM returns one `ExperimentPlan` per round. Python owns
the experiment tree, plan validation, parent-model injection, budget accounting and persistence.
A malformed or illegal plan is rejected and re-asked, never partially applied. State lives on
disk, so a killed run resumes from where it stopped rather than from what a dialog remembers.

**The evaluation protocol is frozen, and configuration — not an agent — decides it.** Chat
template, item count, concurrency, judge routing and runner backend are declared in
`configs/tasks/<task>/eval/protocol.yaml` and read by the evaluator itself. The chat template is
resolved from the model's own `config.json`, never from a caller-supplied label, because a label an
agent can choose is a protocol an agent can move. Scores carry their caliber (`74.00@200/1319`)
everywhere they are displayed, so a subset score can never be read as a full-benchmark one.

**Every experiment produces one typed report, and the next round reads it whole.** `report.json` is
shaped by the four stages that produced it (data / train / eval / review), each carrying its own
status, the workarounds it needed, and the numbers it measured — actual epochs, final train loss,
the chat template training really used, the metric's standard error. There is no hand-picked digest
in between, because a projection's key names can silently stop matching the contract.

**Three peer roles, one orchestrator.** A **leader** plans and reviews; an **experimenter**
implements one node end to end; an **investigator** profiles the pool and reads the scorer's parser
before round 0. None of them can start another — the Python loop invokes all three.

## Architecture

```
┌─────────────────────────── agent host ───────────────────────────┐
│                                                                  │
│   leader ──── plan ────▶ loop: validate · grow tree · budget      │
│     ▲                              │                             │
│     └──── review ◀── report.json ──┤                             │
│                                    ▼                             │
│                              experimenter ─────┐                 │
│                              investigator ─────┤                 │
└────────────────────────────────────────────────┼─────────────────┘
                                                 │ ssh + Slurm
                                                 ▼
┌──────────────────────────── GPU cluster ─────────────────────────┐
│  data operators  ·  training (frozen venvs)  ·  vLLM evaluation  │
└──────────────────────────────────────────────────────────────────┘
```

The agent host runs the agents and owns all durable state. The cluster does every GPU-bound step
inside frozen environments declared in `configs/environments/<env>/env.yaml`; the repo is deployed
there as a git checkout, so deploying is `git pull` (plus `scripts/sync_envs.py` when a tool version
changed).

A **live monitor** ships with the loop (`src/dataos/agent/monitor.py` + `web/`): the experiment
tree, the running node's stage stepper, a console of each agent's reasoning and tool calls, and a
stall alert when every transcript goes quiet. One monitor serves every run under a workspace.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11 on the agent host.
- A code-agent CLI authenticated on the agent host — `claude`, `kimi`, `codex` or `qwen`, chosen
  per run.
- Passwordless SSH to a Slurm cluster holding the frozen environments, the data pool, the base
  models and the benchmark cache.

The agent host itself needs no GPU: it runs agents and bookkeeping, and every GPU-bound step is
submitted to the cluster.

## Configuring a run

One JSON per run, carrying **facts only** — never strategy:

```jsonc
{
  "goal": "Improve Qwen3-1.7B-Base on gsm8k within 10 GPU-hours, using the fixed pool.",
  "workspace": "workspace/ptb-gsm8k",   // the workspace path IS the run identity
  "environment": "PTB_H100_slurm",      // configs/environments/<env>/env.yaml
  "task": "posttrainbench",             // configs/tasks/<task>/
  "bench": "gsm8k",                     // the single target benchmark
  "inputs": {
    "base_model_path": "/data/.../Qwen3-1.7B-Base",
    "fixed_pool": "/data2/.../posttrainbench_fixed_pool_v2"
  },
  "exec": {
    "deploy_root": "/data/.../dataos",  // the repo checkout on the cluster
    "runs_base": "/data2/.../runs",     // per-experiment scratch on the cluster
    "resource": { "gpus": 1, "cpus": 8 }
  },
  "budget": { "time_hours": 10, "eval_samples": -1 },   // -1 = full benchmark
  "agent": { "provider": "claude", "idle_timeout_s": 1800 },
  "monitor_port": 8765
}
```

```bash
uv run dataos configs/runs/ptb-gsm8k.json            # fresh instance
uv run dataos configs/runs/ptb-gsm8k.json --resume   # continue an existing workspace
```

Every field is specified in [`configs/CONTRACT.md`](configs/CONTRACT.md) (§9 covers the run JSON).
Strategy is never configured — it is the leader's per-round decision, persisted in the workspace.

## How a run proceeds

1. The **investigator** profiles the fixed pool and reads the scorer's parser, then evaluates the
   base model to establish the baseline. Both land in the workspace before any training.
2. Each round, the loop hands the leader the tree and the recent reports; it returns one
   `ExperimentPlan`. The loop validates it, adds a child node and injects the parent's model path.
3. The **experimenter** realizes the plan on the cluster and writes `report.json` **incrementally** —
   the score reaches disk the moment the evaluator's `summary.json` exists, so a CLI that dies
   mid-analysis loses the review, never the measurement.
4. The leader reviews; the loop attaches the score and grows the tree. Slurm queue waiting is
   measured and excluded from the budget, so contention costs wall-clock but not experiments.
5. When the budget is spent, the best node's model is recorded in `final_model.json`.

Failure handling is part of the loop rather than bolted on: dropped CLI streams reconnect and
resume the same session, a silent agent is killed and resumed after `idle_timeout_s`, and a run
whose evaluation completed but whose report was never finalized has its score salvaged from disk.

## Repository layout

A single [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/); no external git
pins.

| Path | Contents |
|---|---|
| `src/dataos/` | The loop, planner and reviewer, role runners, the frozen-eval boundary, the `Project → Study → Experiment → Stage` model, the report schema, prompts and the monitor |
| `src/dataos_agent/` | The code-agent CLI driver: a neutral event stream over four providers, transport self-healing, and a stall watchdog |
| `tools/train/` | **dataos-train** — deterministic training lifecycle (validate / build / submit / status / collect) over LLaMA-Factory and verl |
| `tools/operator/` | **dataos-operator** — 52 data-recipe operators (27 selectors, 10 evaluators, 10 mappers, 4 mixers, 1 reweighter) |
| `tools/eval/` | **dataos-eval** — submits and collects one frozen benchmark evaluation, resolving the protocol from config plus the model's own `config.json` |
| `configs/` | Environments, tasks (benchmarks, eval protocol, vendored benchmark data) and run definitions |

Prompts are per role and per purpose: `src/dataos/prompt/<role>/context/` holds what a role always
knows, `<role>/turn/` holds what it is asked this turn. Tool versions on the cluster are pinned by
the `dataos_tools` declaration in `env.yaml`, kept equal to each `pyproject.toml` by a test and
applied with `scripts/sync_envs.py`.

## Evaluation protocol

The PostTrainBench implementation is self-contained in `configs/tasks/posttrainbench/eval/`, with
benchmark data vendored under `configs/tasks/posttrainbench/benchmarks/` so a run needs no network.
Five benchmarks (GSM8K, HumanEval, BFCL, GPQA-main, AIME 2025) are scored inside the GPU job; two
judge benchmarks (HealthBench, ArenaHard-writing) generate in the job and are scored afterwards by
an OpenAI-compatible judge. Protocol decisions and their evidence live in `docs/eval/`.

## Development

```bash
uv run ruff check src tests
uv run pytest -q
```

The suite is fast (no GPU, no network) and is the contract between the layers: it pins the report
schema against the prompt that asks for it, the training spec against `dataos-train`'s own
validator, the declared tool versions against each package, and the eval protocol against the
values the evaluator resolves.

`AGENTS.md` is the operating guide for AI coding agents working in this repository — conventions,
invariants and hard-won cluster facts. Human contributors will find it useful too.

## Documentation

| Document | Contents |
|---|---|
| [`configs/CONTRACT.md`](configs/CONTRACT.md) | Environment / task / run-JSON specification |
| [`AGENTS.md`](AGENTS.md) | Conventions and invariants for agents editing this repo |
| [`tools/train/README.md`](tools/train/README.md) | TrainSpec, backends, online operators |
| [`tools/operator/README.md`](tools/operator/README.md) | The operator catalog |
| `docs/eval/` | Eval protocol decisions and baseline results |
| `docs/dataos_agent/` | The CLI driver layer |
| `docs/troubleshooting.md` | Operational failure modes and their fixes |

## Status

Alpha (`0.3.0a0`). The loop, the tools and the PostTrainBench task are in daily use; interfaces
between them are stable enough to build on but are not yet frozen. Issues and pull requests are
welcome — please run `ruff` and `pytest` first, and describe what a change is meant to prevent
rather than only what it does.

## License

MIT — see [`LICENSE`](LICENSE).
