# AGENTS.md

Operating guide for AI coding agents working in this repository. Read it before changing
shared code, and keep these invariants unless the user explicitly asks to change them.

## What this repo is

`DataOS` is an **autonomous PostTrainBench agent**. It post-trains ONE base model to maximize ONE
PostTrainBench benchmark, within a fixed data pool and a fixed GPU-time budget.

- A persistent **leader** agent (a code-agent CLI dialog) maintains an experiment **tree**. Each
  round it returns ONE `ExperimentPlan` (parent node + data recipe + train config) and ONE
  `ReviewOut`. It only proposes plans / reviews — it never executes, edits files, or writes model
  paths.
- **Code** (`dataos.agent.loop.run_auto`) owns the tree, plan validation, parent-model injection,
  frozen eval, and persistence.
- An **experimenter** agent (also a code-agent CLI, with a shell) realizes ONE experiment end to
  end on the H100 cluster: build SFT data (DataOS-Operator) → train with LLaMA-Factory (Slurm) →
  evaluate with the lm-eval PostTrainBench protocol → write `report.json`. (A third role, the
  **investigator**, gathers pre-training intel once per run and writes `recon.json`.)
- Both agents run on the **agent host**; all GPU work ships to the **H100 cluster** over SSH.

There is exactly ONE execution path: the agentic experimenter. The old in-process "code-executor"
(`run_one.py`, `runtime/executor.py`, in-process train/eval stages) was removed — do not reintroduce it.

## Stack

- Python `>=3.11`; env/build via `uv` + `setuptools`; lint `ruff`; tests `pytest`.
- Core deps: `datasets`, `pyarrow`, `pydantic`, `tenacity`, `dataos-operator` (operator catalog),
  `dataos-agent` (code-agent CLI drivers: claude/kimi/qwen + transport self-healing; git-tag dep).
  Training/eval stacks (LLaMA-Factory, lm-eval, vLLM, torch) live in **frozen uv venvs on H100**
  (`sft_env`/`rl_env`/`eval_env`, see configs/environments), not the agent-host venv.
- LLM backend: a **code-agent CLI** (`backend="cli"`) driven via the **dataos-agent** package
  (`cli_agent.py` adapter); provider/model are config facts (`agent_provider` default `"kimi"`,
  `agent_model` default `None` = the CLI's own default model), auth is the CLI's own login on
  the agent host. Transport self-healing (endpoint drop → backoff → resume-reconnect) lives
  INSIDE dataos-agent.

## Repository map

- `src/dataos/agent` — the loop: `loop.py` (`run_auto`), `state_tree`, `planner`, `reviewer`,
  `subagent_executor` (drives the per-experiment experimenter + the investigator), `eval_freeze`,
  `schemas`,
  `stages` (the data→train→eval contract; each stage's `invoke` is assembled from registered
  tools), `tools` (the pluggable stage-tool registry: Operator/Trainer/Evaluator protocols —
  the Trainer is `DataosTrainCli`: the deterministic dataos-train launcher lifecycle
  validate/build/submit/status/collect; its CLI lives in the thin `train_cli_env`),
  `run_config` (the single-JSON config boundary), `backend`, `budget`, `config`, `logging`,
  `llm/` (`cli_agent.py` — the provider-neutral dataos-agent adapter — + structured-output
  helper), `posttrainbench.py`
  (the eval protocol: bench list + sampled limits per profile), and `monitor.py`
  (the live HTTP monitor — pure stdlib, embedded by `run_auto`; serves the `web/` UI).
- `src/dataos/prompt` — EVERY prompt as a Jinja2 YAML template, in THREE layers/folders:
  `system/` (role skeletons: `leader` / `experimenter` / `investigator` + `strategy`),
  `kb/` (single-sourced knowledge blocks assembled into system prompts: operator_* / pool_survey /
  training / cluster_discipline / eval_format / report_contract / failure_diagnosis + the three
  stage-tool blocks), `task/` (per-call task messages — facts only). No prompt text is hardcoded
  in `.py`; render via `from dataos import prompt; prompt.render(name, **facts)` where `name` is
  the folder-relative path (`prompt.render("kb/cluster_discipline")`). System prompts are composed
  in ONE place — `backend.system_prompt(role, cfg)` (`_BLOCKS`: skeleton + kb blocks, joined with
  blank lines); run facts (budget, GPUs) are render params of the leader skeleton only.
  `system/strategy.yaml` is a maintainable **strategy playbook** appended to the leader system
  prompt; update it from each run's trajectory analysis.
- `web/` — the live monitor's single-page UI (`index.html`) and a standalone CLI wrapper
  (`server.py`, `python web/server.py` to browse run history). One monitor per workspace serves
  every run on one port; see "Live monitor" below.
- `src/dataos/core` — the `Project → Study → Experiment → Stage` / `Data·Model·Eval` abstraction,
  `workspace`, `io`, `ids`, `report`, `provenance`, `catalog`, `loading`. (Kept as the data model;
  the `Stage.run()` train/eval executors were removed — execution is the experimenter's.)
- `configs/` — the unified, in-repo config (DataOS-Runtime was merged in and archived):
  `environments/<env>/env.yaml` (deployment facts: access/slurm/frozen uv venvs/paths/gateway/
  ops + the `train:` backend↔env binding), `tasks/<task>/` (task contract + the self-contained
  `eval/` lm-eval bridge + `benchmarks/` data, ~16M git-tracked), `runs/*.json` (per-run facts
  ONLY + required `environment`/`task` pointers — paths/train/env facts are forbidden there and
  rejected at load). `configs/CONTRACT.md` is the field spec. The composer
  (`agent/runtime_compose.py`) resolves these into the experimenter's ENV OPS/TASK prompt sections.
- `resources/` — `custom_operators/` (shared sedimented ops), `recon/` (source profiler),
  `data/` (misc).
- `docs/` — architecture notes; **several predate the agent-version refactor and are stale**.
- `tests/` — `pytest`, grouped by agent / core.

## Live monitor

`run_auto` embeds a one-port HTTP monitor (`monitor.py`, default `8765`, config `monitor_port`,
`DATAOS_MONITOR_PORT` overrides, `0` disables). ONE monitor per workspace serves EVERY run: each
run writes a heartbeat `workspace/<run_id>/live.json` (every ~20s + on phase change), and the
monitor classifies a run live/done by heartbeat freshness (killed runs go stale; finished runs
write `phase:"done"`). The host self-heals — any run re-binds the port if the host dies. The page
follows the selected run: a pan/zoom experiment tree, a live status strip with a
`data → train → eval` sub-stage stepper (ssh-probes the running node's H100 `run_root` markers),
and a bottom agent console streaming each agent's reasoning + tool calls (read from Claude Code's
JSONL transcripts under `~/.claude/projects`). Cluster ssh creds reach the monitor via
`monitor.configure(cluster)` and are kept OUT of the served `LiveState` (no secret leak).

## Local commands

```bash
uv sync                          # agent-host env
uv run dataos configs/runs/ptb-h100-qwen-gsm8k.json   # start a run
uv run ruff check src tests
uv run pytest -q
```

Note: `uv sync` may need the proxy OFF and a domestic mirror (the proxy is for reaching the
Claude/gateway endpoints, and breaks TLS to domestic PyPI):
`unset HTTPS_PROXY HTTP_PROXY ALL_PROXY; UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync`.

## H100 cluster facts

GPU work runs on the lab H100 Slurm cluster; the agent host reaches it over SSH.

- SSH: `Host h100` → `HostName 114.214.215.159`, `User hcgu`, `IdentityFile ~/.ssh/ghc_miyao`.
- Deploy (the repo on H100): `/data/hcgu/users/tjh/dataos_agentver` — a **git checkout of `origin/main`**.
  Deploy = `git push origin main` (host) then `ssh h100 'git -C <deploy_root> pull'`.
- Run scratch (per experiment): `/data/hcgu/users/tjh/dataos_agentver_runs/...`.
- Fixed pool: `/data2/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v2`
  (env.yaml `paths.fixed_pool`). Eval benchmark data ships IN the repo
  (`$AGENTVER/configs/tasks/posttrainbench/benchmarks` — comes down with `git pull`).
- Frozen uv venvs (defined in `configs/environments/PTB_H100_slurm/envs/`): `sft_env`
  (LLaMA-Factory + torch), `rl_env` (verl), `eval_env` (lm-eval + vLLM), under
  `/data/hcgu/users/tjh/dataos_runtime/environments/PTB_H100_slurm/envs/`.
- Submit GPU work through Slurm (default `gpu04`); the login node is orchestration only.
- **Compute nodes have no internet.** Datasets resolve offline from the benchmarks HF cache; the
  2 judge benches generate on the GPU node then score on the login node.

## Invariants to respect

- **Config = facts, not strategy.** `configs/runs/*.json` holds paths / the single bench / budget /
  resources / `monitor_port` (host-side infra) only. Data recipe, training method (LoRA vs full,
  hyperparameters), tree growth, and eval sampling are the **leader's** decisions, persisted in
  per-round plans. Do not add *strategy* knobs to the config or CLI (`monitor_port` is a fact, not
  strategy — it's fine).
- **Training = LLaMA-Factory, authored by the experimenter.** The experimenter writes the LF YAML/commands
  per the PLAYBOOK; there is no in-repo bespoke trainer.
- **Eval = the `lm_eval_bridge`** (pre-main lm-eval protocol; baseline-comparable). Do not reintroduce
  the InspectAI/external-scorer path.
- **Secrets never enter git.** Gateway/judge API keys live in `0600` key_files on H100, exported
  inline at run time. The config carries `base_url` / `model` / `key_file` paths only.
- The leader returns JSON only (validated; it is re-asked on invalid output); it never writes
  model paths or changes the eval.
- **Prompts live in `src/dataos/prompt/*.yaml`, never hardcoded in `.py`.** Add/edit a prompt by
  editing its YAML template; pass run-specific facts as `prompt.render()` params (no literals like
  "10h" / "60000" baked into the text). `system/strategy.yaml` is appended to the leader prompt as
  empirical *priors* — it guides decisions but the agent still chooses each plan; keep it in YAML
  and maintain it from trajectory analysis, don't turn it into hard config knobs.

## Coding style

- **Do not over-engineer ("如无必要，勿增实体").** Prefer one path over a path + fallback; no
  speculative abstractions, compat aliases, or config knobs the single PTB use case doesn't need.
  When unsure, make the smaller change.
- Follow the existing style; keep lines ≤ 100 (ruff). Prefer small explicit helpers; sparse comments
  only where logic is non-obvious. ASCII unless the file already needs Unicode.
- Preserve the `dataclass` / type-hint patterns in `src/dataos/core` and `src/dataos/agent`.

## Validation

Before finishing a meaningful change: `uv run ruff check src tests` and `uv run pytest -q`. Run a
relevant smoke command when the change affects runtime behavior (e.g. an H100 eval/train job).

## Useful entry points

- Run entry: `src/dataos/agent/cli.py` → `dataos.agent.loop.run_auto`.
- Config boundary: `src/dataos/agent/run_config.py`.
- Experimenter/investigator driver: `src/dataos/agent/subagent_executor.py`.
- Eval: `resources/code/posttrainbench/lm_eval_bridge/`.
- Prompts: `src/dataos/prompt/*.yaml` (loader: `src/dataos/prompt/__init__.py`).
- Strategy playbook: `src/dataos/prompt/strategy.yaml` (appended by `backend._system_prompt`).
- Live monitor: `src/dataos/agent/monitor.py` + `web/index.html` (standalone: `web/server.py`).
