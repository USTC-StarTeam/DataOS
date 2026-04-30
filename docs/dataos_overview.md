# DataOS — Project Overview

> Living document. Snapshot of where the project stands, where it is going, and how
> the pieces are designed to fit together. Anyone joining the project should read
> this first.

Last updated: 2026-04-30.

---

## 1. What we are building

A **Research Operating System** for data-centric auto-research on post-training.
Not "another coding agent", not "another harness on top of Claude Code". A
**control plane** with its own native daemon agent, designed so that:

- Long-running auto-research campaigns (think 10+ hours, unattended) are
  first-class.
- Multiple kinds of work happen **concurrently**: idea generation, experiment
  execution, result analysis, candidate-operator design, reporting.
- State is persistent, auditable, and recoverable from crashes.
- Both our own agent (DataOS Agent) and external coding agents (Claude Code,
  Codex) can drive the same OS through the same contracts.

The single sentence definition:

> **DataOS** is a control plane for auto-research workloads, with a default
> *daemon-shaped* agent that runs alongside *shell-shaped* agents (CC, Codex)
> through skills. It is agent-agnostic and built around a persistent,
> event-sourced state machine.

---

## 2. Goals (the next ~4 weeks)

We are optimizing simultaneously for three things, in this priority order:

1. **EMNLP submission** — paper deadline ≈ 4 weeks out.
2. **PostTrainBench leaderboard entry** — the bench evaluates a coding agent
   that, given 10h of H100 time, must turn a base model into the strongest
   post-trained model it can. DataOS is a natural fit: its job is exactly to
   orchestrate auto-research under a hard time/compute budget.
3. **Open-source ecosystem** — make the OS the substrate that any coding agent
   can plug into. CC and Codex users adopt us via a few skills; researchers who
   want a daemon get DataOS Agent.

These three goals are aligned, not competing: the paper uses PostTrainBench as
the empirical anchor, and the ecosystem story (multiple agents on one OS) is
itself a contribution.

### Paper framing

Working title direction: *DataOS: a Research Operating System for Auto
Post-Training*.

Argument:

1. PostTrainBench-style "agent + 10h H100 → best base model" tasks demand a
   **daemon-shaped agent** (long-lived, concurrent, state-machine-driven,
   budget-aware), which structurally cannot be a CC-style **shell-shaped**
   agent.
2. We propose DataOS: a control plane (state, scheduler, lease, budget, DAG,
   candidate-operator gate) + a reference daemon agent + adapters that let
   CC/Codex drive the same OS as clients.
3. On PostTrainBench, all three agent forms (DataOS Agent / CC + skill /
   Codex + skill) running on the same OS beat single-agent baselines. The
   throughput comes from the OS, not the LLM.
4. Ablations: remove concurrency, remove backpressure, remove candidate gate,
   single-loop manager, swap policies, etc.

---

## 3. Current state (snapshot)

### 3.1 Code already in repo

Under `dataos/research/` (control plane, ≈ 3.7k LoC):

| File | Role |
|------|------|
| `models.py` | Dataclasses + enums (`ResearchProject`, `ResearchTask`, `Resource`, `ResourceLease`, `Hypothesis`, `CandidateOperator`, `AgentTask`, `AgentProposal`, `Event`) and id factories. |
| `store.py` | SQLite persistence. Single source of truth (`<workspace>/dataos.db`). All state transitions, validations, and side-effects live here. |
| `scheduler.py` | `scheduler_tick`, `recover_stale_tasks`, resource selection, default local resource bootstrap. |
| `worker.py` | `worker_run` for `stub_result`, `experiment_run` (via `runtime.run_task`), `operator_validation`, and SSH-assigned tasks. |
| `agent.py` | Current `DataOSAgent` (manager + parallel specialists), `accept_agent_proposal`, `reject_agent_proposal`, role-based prompt scaffolding. |
| `orchestrator.py` | `run_ready_tasks`, `run_project_step` (pull-mode coordination of agent + scheduler + worker). |
| `api.py` | Flat function surface that CLI / external callers consume. |

CLI (`dataos/cli.py`, ≈ 600 LoC, argparse only): subcommand groups for
`project / task / resource / candidate / scheduler / worker / agent / event`.

Tests: `tests/test_research_os.py` (54 tests covering candidate gate, budget,
project lifecycle, pause-release-lease, cancel cascade, completion guard, DAG
block/unblock, atomic schedule claim, requeue, stale recovery, parallel
specialists, report gating, proposal validation, CLI E2E).

Docs in repo: `docs/research_os_v1.md` (user-facing), `docs/system_implementation.md`.

### 3.2 What the control plane already enforces

These properties are real and tested today, not aspirational:

- **Project lifecycle**: `created → running → paused / completed / cancelled`,
  with terminal states irreversible. `created → running` is auto-triggered by
  any work submission. `paused` requeues all `scheduled` tasks (releases
  leases). `cancelled` cascades to open tasks and agent tasks. `completed`
  requires zero open work. Report proposals updating the project summary go
  through the same guard.
- **Research task DAG**: dependencies are first-class. Failed/cancelled/blocked
  upstream propagates to `blocked` downstream on the next scheduler tick.
  Recovery (requeue) of upstream un-blocks downstream automatically.
- **Budget**: `max_runs` / `max_experiments` cap non-cancelled
  `experiment_run` tasks. `accept_agent_proposal` pre-reserves budget per
  proposal item so a batch of new tasks cannot collectively overshoot.
- **Project concurrency**: `max_concurrent_tasks` / `max_concurrent_runs`
  caps `scheduled+running` per project so one campaign cannot starve others.
- **Atomic scheduling**: `schedule_research_task` claims a task with
  `WHERE status='pending' AND lease_id IS NULL` and asserts `rowcount==1`.
- **Resource/lease**: resources have `host`, `gpu_ids`, `tags`,
  `max_concurrent_tasks`. Leases are append-only with `active/released`. Task
  `resource_request` matches by gpus / host / tags. A default
  `local-default` resource is auto-injected (0 GPU).
- **Candidate operator gate**: `experiment_run` may reference a candidate
  operator only after status is `validated` or `promoted`. Validation tasks
  write status back. Stable operators are not gated.
- **Proposal contract**: proposals must belong to the same project and role
  as their source agent task. Accept requires an active project; reject is
  allowed on inactive projects (for human cleanup). Accepted/rejected/
  superseded are terminal.
- **Terminal task retry**: `update_task_status` cannot move a terminal task
  back to active states; retries must go through `requeue_research_task`,
  which re-checks budget and project status.
- **Stale recovery**: `scheduler recover-stale` requeues `scheduled/running`
  tasks older than a cutoff with proper lease release.

### 3.3 What is *not* yet in the code

- No daemon. Today everything is pull-mode: a CLI invocation runs one tick
  (`scheduler tick`, `agent cycle`, `project step`).
- No supervisor / multi-loop architecture. `run_project_step` runs the agent
  cycle and ready-task run in parallel via a 2-thread pool, but that is one
  shot, not a continuously running daemon.
- No backpressure (queue depth, token budget, wall-clock budget). `budget`
  today only counts experiment runs.
- No `wall_clock_hours` budget. PostTrainBench needs this.
- No `dataset_curator` agent role; today's roles are
  `hypothesis / experiment_planner / operator_designer / evaluation_analyst /
  critic / report` plus the implicit `research_manager`.
- No stable operator library. `dataos/operators/` does not exist yet.
- No `data_prep` flow for pulling external datasets. `task_type` set is
  `experiment_run / eval_run / operator_validation / analysis_task /
  artifact_check`. We can reuse `artifact_check` for now.
- No PostTrainBench harness. `dataos/eval/` has unrelated eval tasks
  (aime2025, livecodebench, openfindata, physics, climaqa, toxicity).
- No skill adapter for CC/Codex (the `skills/` directory exists but is empty
  for our purposes).

---

## 4. Why DataOS Agent exists alongside CC/Codex

CC and Codex are excellent **shell-shaped** agents: interactive, single-loop,
LLM-per-step, session-bound. They cannot be the runtime of a 10h unattended
auto-research campaign. Five structural reasons:

| Property | Shell-shaped (CC) | Daemon-shaped (DataOS Agent) |
|---|---|---|
| Lifetime | One session | Hours/days, survives restarts |
| Concurrency | Single main loop, subagents block parent | 4 independent loops + specialist pool |
| Decision form | Every step is an LLM call | Manager decisions are deterministic state-machine; LLM only inside specialists |
| Resource awareness | Has to be told via prompt every time | Reads control-plane tables; constraints enforced by the store |
| Recovery / fork | Crash = lost session | Recover from events table; campaign can be forked |

CC remains stronger at: writing/editing code, reading unfamiliar repos, talking
to humans, prompt iteration, ecosystem maturity. **DataOS Agent should not
attempt to compete on those.** Code-writing inside a campaign should be
delegated to a CC/Codex worker behind a `task_type`.

The right framing: **two different agent shapes, one OS underneath.**

---

## 5. Architecture (target, post-redesign)

```
┌────────────────────────────────────────────────────────────────┐
│ DataOS Agent Daemon (one per project)                          │
│                                                                │
│   ┌────────────┐                                               │
│   │ Supervisor │   lifecycle, signals, deadlines, backpressure │
│   └─────┬──────┘                                               │
│         │                                                      │
│   ┌─────┴────────┬────────────┬─────────────┬───────────┐      │
│   ▼              ▼            ▼             ▼           │      │
│  Generator    Executor    Analyzer       Reporter       │      │
│   Loop         Loop         Loop          Loop          │      │
│   │             │            │              │           │      │
│   │             ▼            │              │           │      │
│   │     ┌──────────────┐     │              │           │      │
│   └────►│ Specialist   │◄────┴──────────────┘           │      │
│         │  Pool (LLM)  │                                │      │
│         └──────┬───────┘                                │      │
│                │                                        │      │
└────────────────┼────────────────────────────────────────┘      │
                 │ reads/writes                                  │
            ┌────┴─────────────────────────────────────────┐     │
            │ Control Plane (ResearchStore, dataos.db)     │◄────┘
            │  projects · tasks · leases · candidates ·    │
            │  hypotheses · agent_tasks · proposals ·      │
            │  events                                       │
            └──────────────────────────────────────────────┘
                 ▲                ▲                  ▲
                 │                │                  │
            ┌────┴───┐      ┌─────┴────┐      ┌──────┴──────┐
            │ DataOS │      │ CC skill │      │ Codex skill │
            │ Agent  │      │ adapter  │      │  adapter    │
            └────────┘      └──────────┘      └─────────────┘
```

### 5.1 Control plane (already exists)

Persistent, event-sourced state in SQLite. Owns all transitions, validations,
and lifecycle guards. Exposes a flat Python API + CLI. **Agent-agnostic.**

### 5.2 DataOS Agent (to be redesigned into a daemon)

Four independent loops, each implemented as `loop.tick()` running on its own
thread, communicating only through control-plane events.

| Loop | Cadence | Job | LLM? |
|------|---------|-----|------|
| Generator | slow (≥30s) | hypothesis / experiment_planner / dataset_curator / operator_designer / critic spawning, conditional on backpressure | manager: no, specialist: yes |
| Executor | fast (1–5s) | wraps `scheduler_tick` and dispatches `worker_run`; pure scheduling | no |
| Analyzer | medium (≥10s) | spawns `evaluation_analyst` / `critic` for new completed tasks; enforces `covered_task_ids` deduplication | manager: no, specialist: yes |
| Reporter | one-shot triggered | wakes when budget exhausted or deadline near; spawns `report` specialist; enforces final summary write | manager: no, specialist: yes |

**Manager policies are pure functions** of `(state snapshot) → list[Action]`.
Single-tick, deterministic, unit-testable, swappable for ablations. LLM calls
happen only inside the specialist pool, never in the manager.

### 5.3 Specialist Pool

Shared by all loops. Owns LLM concurrency, retry, token-budget tracking, and
cross-loop priority (analyzer > reporter > generator under token pressure).

### 5.4 Backpressure (three layers, supervised)

1. **Queue backpressure** — generator stops when
   `ready+scheduled+running` exceeds a threshold.
2. **Token backpressure** — pool throttles low-priority specialist requests
   when 24h token usage is high.
3. **Wall-clock backpressure** — reporter watches the clock. Inside the
   safety margin: stop generator, drain executor, force a final report.

This is what makes the agent budget-safe under PostTrainBench's 10h ceiling.

### 5.5 Observability

Every loop emits structured events:

| Event | Use |
|---|---|
| `loop_tick` | scheduler load |
| `specialist_call` | LLM cost, latency |
| `queue_snapshot` | concurrency utilization |
| `backpressure_triggered` | enforcement evidence (paper figure) |
| `budget_used` | budget curves (paper figure) |
| `recovery_action` | reproducibility evidence |

These events are the source of paper figures and the ablation tables.

---

## 6. Operators: two-layer model

A correction over earlier drafts: **operators come in two layers**, only the
second goes through the candidate gate.

### 6.1 Stable operators (`dataos/operators/`, owned by the operator engineer)

Hand-written, version-controlled, opinionated, **out-of-the-box**. Used by
the agent without gating: `operator_refs: [{type: "stable", id: "..."}]`.

First-batch target (W1–W2):

| Family | Operator | Notes |
|---|---|---|
| Filter | `quality_filter` | heuristic + LLM judge variants |
| Filter | `length_filter` / `dedup` / `lang_filter` | cheap, broad |
| Synthesis | `instruction_synthesis_template` | template-based, no LLM cost spike |
| Synthesis | `self_instruct_lite` | small LLM-driven |
| Mix | `mixture` | weighted ratio mixer |
| Mix | `curriculum_sort` | difficulty proxy ordering |
| Mix | `category_balance` | stratified balance |

Roughly 8–10 stable operators is enough for a working W2 PostTrainBench loop.

### 6.2 Candidate operators (runtime synthesis)

Produced by the `operator_designer` specialist when stable operators do not
cover what the analyzer/critic is asking for. Tracked in the control plane:

```
proposed → generated → validating → validated → promoted (or rejected/failed)
```

A candidate may be referenced by an `experiment_run` only after `validated`
or `promoted`. Validation is itself a `task_type=operator_validation`
task (CPU lane). Promotion to the stable registry is the natural community
contribution path.

---

## 7. Tasks, resources, and resource lanes

### 7.1 Task types

Today the set is `experiment_run / eval_run / operator_validation /
analysis_task / artifact_check`. For PostTrainBench we add (or alias via
`artifact_check`):

- `data_prep` — pull a dataset, normalize to HF format, `save_to_disk`.
- `dataset_profile` — compute length / language / dedup / domain stats.

Both are CPU/IO bound and must not occupy GPU leases.

### 7.2 Resource lanes

Resources are tagged. The convention (to enforce in scheduler match):

- `tags: ["gpu"]` — training and any GPU-bound eval. `experiment_run`
  defaults to `gpus=1`, this lane.
- `tags: ["cpu", "io"]` — `data_prep`, `dataset_profile`,
  `operator_validation`, judge-based evals. Default `gpus=0`.

Cross-lane scheduling is what lets the daemon keep GPU saturated while data
prep and analysis run in parallel.

### 7.3 Wall-clock budget

`budget.wall_clock_hours` becomes a project-level field, surfaced through
`get_experiment_run_budget`-style summaries. Reporter loop reads it; the
supervisor enforces graceful shutdown before the deadline.

---

## 8. Reference run: PostTrainBench (10h H100, base model)

The full mental model. This is the case study for paper section 1.

| t | What happens |
|---|---|
| 0 | User runs `dataos project create` and `dataos agent daemon run`. Daemon starts. State: 1 project, 0 tasks. |
| 0–3 min | Generator spawns initial 4 specialists in parallel: `hypothesis`, `dataset_curator`, `experiment_planner`, `critic`. |
| 3–5 min | Manager (state machine) ingests proposals. Hypotheses land in store. `dataset_curator` produces a list of dataset references → manager turns into N `data_prep` tasks. `experiment_planner` returns null (deps missing); state machine records "pending deps". |
| 5–30 min | Executor saturates CPU/IO lane on `data_prep` + `dataset_profile`. **In parallel**: generator preps next planner round, analyzer ranks hypotheses via critic, reporter idles. |
| 30–45 min | `experiment_planner` v1 produces 8 candidate experiments. Manager accepts 6 (saves 2 budget for adaptive replanning). |
| 45 min – 1.2h | Executor schedules first `experiment_run` on GPU. Worker calls `runtime.run_task` (recipe → train → eval). Generator hits queue backpressure (in-flight=1, queued=5) and pauses. |
| 1.2–1.5h | First completion. Score lands. Analyzer spawns `evaluation_analyst`. Executor schedules next experiment immediately on GPU release. |
| 1.5–2h | Analyzer surfaces "math up, code down" insight. Critic suggests a curriculum experiment. Generator spawns `experiment_planner` v2 with new context; planner cancels exp_4 in queue and inserts a curriculum exp_9. **Adaptive replanning. CC cannot do this cleanly.** |
| 2h – 9h | Steady state: ~1.2h per experiment, analyzer keeps up, occasionally `operator_designer` proposes a candidate operator (e.g. zh-aware quality filter), validation runs in CPU lane, promoted before next batch. Token / GPU utilization events accumulate. |
| 9h | Wall-clock backpressure triggers: generator stops, executor drains. |
| 9.6h | Last in-flight experiment completes. Reporter spawns `report` specialist; selects best ckpt; writes `submission.json`. |
| 9.9h | `update_project_summary` → `project status = completed`. |
| 10h | Daemon graceful shutdown. State fully recoverable from events table. |

What CC + skill cannot replicate at parity:

- 10h unattended.
- 4 loops working concurrently.
- Adaptive replanning mid-campaign without prompt-context explosion.
- Hard wall-clock enforcement.
- Crash recovery.

This trace **is** the contribution narrative. Every row of the paper table 1
maps back to one of these advantages.

---

## 9. Repo structure (target)

```
DataOS/
├── dataos/
│   ├── research/             # control plane (exists)
│   │   ├── store.py
│   │   ├── scheduler.py
│   │   ├── worker.py
│   │   ├── orchestrator.py
│   │   ├── api.py
│   │   └── ...
│   ├── agent/                # NEW: daemon, loops, policies, specialist pool
│   │   ├── daemon.py
│   │   ├── specialist.py
│   │   ├── policies/
│   │   │   ├── generator.py
│   │   │   ├── analyzer.py
│   │   │   ├── reporter.py
│   │   │   └── executor.py
│   │   ├── loops/
│   │   │   ├── generator.py
│   │   │   ├── executor.py
│   │   │   ├── analyzer.py
│   │   │   └── reporter.py
│   │   └── prompts/
│   ├── operators/            # NEW: stable operator library
│   │   ├── filter/
│   │   ├── synthesis/
│   │   └── mix/
│   ├── eval/                 # exists, extend with posttrainbench harness
│   ├── adapters/             # NEW: worker registry, skill bridges
│   ├── runtime/              # exists (run_task)
│   ├── core/                 # exists
│   ├── llm/                  # exists
│   └── cli.py
├── skills/                   # CC/Codex skill files (skill engineer)
│   ├── dataos-skill/
│   └── dataos-codex/
├── benchmarks/
│   └── posttrainbench/       # submission entry, 10h budget config
├── tests/
└── docs/
    ├── dataos_overview.md          # this file
    ├── research_os_v1.md           # exists
    ├── system_implementation.md    # exists
    ├── agent_design.md             # planned: daemon design detail
    ├── posttrainbench_trace.md     # planned: extracted from §8
    └── contracts.md                # planned: task / role / spec / event schemas
```

---

## 10. Team and ownership

Four contributors, decoupled along the OS / userland boundary. Cross-team
contracts are the only coupling.

| Person | Owns | Outputs | Public surface for others |
|---|---|---|---|
| You | DataOS core: control plane evolution, daemon, scheduler/policies, reference DataOS Agent. | `dataos/research/`, `dataos/agent/`, daemon CLI. | Python API + CLI as documented. Will not break v0.1 surface during the 4 weeks. |
| Operator engineer | Stable operator library; sanity-check templates for candidate validation. | `dataos/operators/` first batch (8–10 ops). | Each operator registered with a stable id and a `(spec → impl → metric)` contract. |
| Eval engineer | Post-training eval, PostTrainBench harness, ablation hooks. | `dataos/eval/posttrainbench/`, `eval_run` worker. | `eval_run` task spec + `result.score` (plus per-category breakdown). |
| Skill engineer | CC and Codex skill packs; worker registry; integration demos. | `skills/dataos-skill/`, `skills/dataos-codex/`, `dataos/adapters/`. | Talks to the OS only via Python API + CLI. |

The four work in parallel because they share **data contracts**, not function
calls.

---

## 11. Timeline

**W1 — current.** Lock contracts, run baseline, scaffold daemon.
- You: freeze v0.1 API surface; sketch `dataos/agent/` daemon skeleton with
  stub loops; write `agent_design.md` and `contracts.md`.
- Operator engineer: 2–3 stable operators (filter family + mixture).
- Eval engineer: PostTrainBench end-to-end baseline (base model → 10h train
  → eval), record numbers, define `eval_run` spec.
- Skill engineer: CC skill v0 — 5–6 skills (create project / submit task /
  list proposals / accept / get state / run ready).

**W2.** First full unattended campaign on PostTrainBench.
- You: daemon runs; wall-clock + queue backpressure; recovery on startup.
- Operator engineer: +2–3 operators across remaining families.
- Eval engineer: ablation hooks; per-category breakdown.
- Skill engineer: Codex skill + adapter; demo script and README.

**W3.** Three agents on one OS; ablation suite.
- You: CC entry and Codex entry can drive the daemon (or run alongside it
  through the same store). Begin paper §3–§4.
- Operator engineer: tune existing operators; documentation.
- Eval engineer: ≥3 ablation groups, multi-seed; tables.
- Skill engineer: documentation, demo video script.

**W4.** Writing, reproducibility, slack.
- All: paper sections, reproducibility scripts, last-mile experiments. Final
  3 days frozen for proofreading and emergencies.

---

## 12. Risks and what to watch

- **PostTrainBench submission shape unknown** (script vs container vs API,
  base model choice, time accounting). This must be resolved in W1; otherwise
  W2 cannot start.
- **GPU availability**. We need to be sure we can afford at least one true 10h
  H100 run by mid-W2, ideally two.
- **API surface drift**. v0.1 freezes this week. Any change after W1 forces
  cross-team rework and risks the deadline.
- **Specialist prompt quality**. The current prompts are intentionally
  minimal; they must be hardened before W2 to avoid losing campaign hours to
  malformed proposals.
- **Adaptive replanning correctness**. The §8 t=1.5h moment is where the
  daemon really earns its keep. Must be tested with synthetic events before
  W2 to avoid runtime surprises.
- **Two-layer operator confusion**. Make sure the agent prefers stable
  operators by default and only emits candidates when justified, otherwise
  validation tasks will eat the CPU lane.

---

## 13. Working agreements

- **State is truth.** No long-lived in-memory state in agents or workers.
  Everything reconstructs from `dataos.db`.
- **Manager decisions are pure functions** of state. LLM only inside
  specialists. This is both a design rule and the line we will defend in the
  paper.
- **No new feature in the control plane unless a campaign needs it now.**
  W1–W4 favors stability over surface area.
- **Events are append-only.** Every behavior worth analyzing is an event.
- **Two-layer operators.** Stable for production; candidate for runtime
  experiments behind the gate.
- **Never break v0.1 surface during the deadline window.**

---

## 14. Pointers

- User-facing CLI walkthrough: `docs/research_os_v1.md`.
- Implementation details (existing code): `docs/system_implementation.md`.
- Daemon design (planned, this week): `docs/agent_design.md`.
- Reference run (planned, this week): `docs/posttrainbench_trace.md`.
- Contracts (planned, this week): `docs/contracts.md`.
