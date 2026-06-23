# AGENTS.md

## Purpose

This repository is `DataOS`, a local-first, single-experiment runtime for SFT
data workflows.

The runtime is intentionally narrow:

- Internal dataset flow is standardized on Hugging Face `Dataset` and
  `DatasetDict` directories created by `save_to_disk`.
- The LLM is used for planning and review only.
- Operators execute deterministically inside a local sandbox and emit typed
  artifacts.
- Experiments persist reports, plans, reviews, logs, and archived artifacts to
  disk.
- There is no research layer, daemon loop, project queue, or persistent agent
  control plane.

Agents working in this repository should preserve those constraints unless the
user explicitly asks to change them.

## Stack

- Python `>=3.11`
- Package/build tooling: `uv`, `setuptools`
- Linting: `ruff`
- Core dependencies: `datasets`, `pyarrow`, `prompt_toolkit`, `pyclean`

## Repository Map

- `src/dataos/agent`
  - Main-agent loop, state tree, node selection, planner/reviewer prompts,
    schemas, budget, and CLI
- `src/dataos/core`
  - Artifact types, project/study/experiment objects, workspace layout,
    reports, ids, IO, provenance, and local persistence
- `src/dataos/runtime`
  - Experiment execution helpers: local fan-out, sub-agent executor, shared
    data prep, frozen eval selection, and report collection
- `src/dataos/integrations`
  - External integrations for Claude Code, PostTrainBench, LLaMA-Factory,
    Slurm, and lm-eval
- `configs`
  - Runtime and experiment configs, including `configs/runtimes/h100.json`
- `resources/data`
  - HF datasets and benchmark data resources
- `resources/code`
  - Benchmark/harness code resources consumed by runtime and sub-agents
- `scripts`
  - Thin command-line helpers, especially `scripts/run_one.py` and
    `scripts/doctor.py`
- `docs`
  - Architecture notes and runbooks, with the main overview at
    `docs/overview/dataos_overview.md`

## Local Commands

Install dependencies:

```bash
uv sync --extra dev
```

Run lint:

```bash
uv run ruff check .
```

## Remote GPU Hosts

For heavy training or evaluation work that exceeds local capacity, these SSH
hosts are available and accept key-based login directly:

- `bdaa_5090-1`
  - `HostName 114.214.255.119`
  - `User hcgu`
  - `Port 1108`
  - Verified on `2026-03-28`: reachable by SSH and exposes
    `3 x NVIDIA GeForce RTX 5090`
- `bdaa_5090-2`
  - `HostName 114.214.255.119`
  - `User hcgu`
  - `Port 1109`
  - Verified on `2026-03-28`: reachable by SSH and exposes
    `2 x NVIDIA GeForce RTX 5090`

Useful checks:

```bash
ssh bdaa_5090-1 'hostname && nvidia-smi -L'
ssh bdaa_5090-2 'hostname && nvidia-smi -L'
```

### Lab H100 Runtime

PostTrainBench-style training runs should use the lab H100 Slurm cluster when
the user asks for H100 execution.

- SSH alias: `hcgu_h100`
- Login user: `hcgu`
- Persistent workspace root: `/data/hcgu`
- Default DataOS agent-run root:
  `/data/hcgu/dataos_agent_runs`
- Fixed-pool processed root:
  `/data/hcgu/dataos_fixed_pools/processed`
- PostTrainBench containers:
  `/data/hcgu/posttrainbench_containers`
- Base-model cache:
  `/data/hcgu/hf_cache`

Important operating rules:

- The login node is for orchestration only; do not run GPU training or full
  evaluation directly on it.
- Submit GPU work through Slurm, and prefer `gpu04` unless the user explicitly
  chooses another node.
- Keep each agent experiment under
  `/data/hcgu/dataos_agent_runs/<setting>/<coding_agent>/<base_model>/<task>/<run_id>`.
- Do not store tokens, API keys, or private keys in run manifests.
- For fixed-pool PostTrainBench experiments, training data must come only from
  the DataOS fixed-pool manifest recorded in the run directory.

Agent guidance for remote execution:

- Prefer `bdaa_5090-1` for jobs that benefit from 3 GPUs; use `bdaa_5090-2`
  for 2-GPU workloads or to split experiments across machines.
- Treat these hosts as execution backends only. Keep DataOS artifacts, reports,
  plans, and reviews durable in the repository workspace or copy them back
  before claiming a run is complete.
- Do not assume remote environments match local dependencies; verify Python,
  CUDA, and required training stacks before launching long jobs.

## Runtime Facts You Must Respect

### Dataset Artifacts

- `DatasetArtifact` only supports Hugging Face datasets saved on disk.
- Do not introduce alternate dataset storage formats unless the task explicitly
  requires it.
- If code materializes a dataset artifact, refresh or preserve metadata
  consistently with `DatasetArtifact.save()` /
  `refresh_metadata_from_disk()`.

### Sandbox Layout

`Sandbox.prepare()` creates and relies on these directories:

- `inputs/`
- `outputs/`
- `artifacts/`
- `logs/`
- `tmp/`
- `code/`

Reports and review artifacts are expected under the sandbox root, especially:

- `plan.json`
- `report.json`
- `review.json`

### Artifact Model

Supported artifact types are fixed in `src/dataos/core/artifact.py`:

- `dataset`
- `model`
- `eval`
- `result`
- `experiment`

If you add a new artifact type, update the central type definitions and the
local artifact store behavior together.

### Default Runtime

The operator catalog is provided by the external `dataos_operator` package and
surfaced through `src/dataos/core/catalog.py`. Operators are deterministic data
operators only.

`train_llamafactory` and `eval_model` are fixed runtime stages. They are called
directly by `run_task()` after the agent-planned data-operator steps complete.
Do not register them in the operator registry.

### LLM Boundary

The LLM client is for structured plan/review generation. Runtime execution is
still performed by local code.

If no custom client is passed, `run_task()` expects one of:

- `DASHSCOPE_API_KEY`
- legacy `STARTEAM_API_KEY`

Optional overrides:

- `DASHSCOPE_BASE_URL` or legacy `STARTEAM_BASE_URL`
- `DASHSCOPE_MODEL` or legacy `STARTEAM_MODEL`

Keep this separation intact. Avoid pushing execution semantics into prompts
when local runtime code should own them.

## Change Guidance

### When Adding Or Changing An Operator

- Implement against the `dataos_operator` operator interface.
- Accept `inputs` plus `sandbox`, and emit typed artifacts.
- Prefer deriving from the latest relevant upstream artifact in `inputs`.
- Write stable logs via `sandbox.log(...)`.
- Persist operator outputs inside sandbox-managed artifact directories.
- Register the operator in `build_default_registry()` if it should participate
  in the standard runtime.
- Do not add pytest coverage by default; this repository no longer maintains a
  pytest suite.

### When Changing Planning Or Review Behavior

- Keep payloads JSON-structured and validated.
- Preserve fallback behavior when LLM output is invalid.
- Make sure experiment state still writes durable `plan.json` and `review.json`
  files.

### When Changing Artifacts Or Experiments

- Preserve on-disk compatibility where practical.
- Ensure `save_meta()` output remains JSON-serializable.
- Keep parent-child artifact lineage intact.

## Validation Expectations

Before finishing a meaningful code change, run lint and any direct smoke command
that is actually relevant to the change.

Minimum expectation:

- `uv run ruff check .`

Do not recreate the old pytest suite unless the user explicitly asks for it.

## Coding Style

- Follow the existing Python style in the repository.
- Keep lines within Ruff's configured limit of 100 where practical.
- Prefer small, explicit helpers over implicit magic.
- Preserve type hints and dataclass usage patterns already used in
  `src/dataos/core`.
- Keep comments sparse and only where they clarify non-obvious logic.
- Prefer ASCII unless the file already requires Unicode content.

## Safe Agent Behavior For This Repo

- Read the surrounding module before editing shared runtime code.
- Avoid destructive cleanup commands against user data or workspace artifacts
  unless explicitly requested.
- Do not silently change task template schemas, artifact directory conventions,
  or operator names without updating all dependent code paths.
- If a requested change conflicts with the repository's local-first /
  fixed-runtime design, call that out and make the smallest coherent change
  possible.

## Local Feishu / Lark CLI Note

- When operating on the user's Feishu / Lark resources from Codex Desktop, do
  not rely on sandboxed execution.
- In this environment, sandboxed threads may fail to access the local keychain
  used by `lark-cli`, which can surface as `keychain not initialized` even when
  the user has already authorized successfully.
- Prefer running Feishu / Lark CLI commands outside the sandbox using the direct
  Node entrypoint:

```bash
/opt/homebrew/bin/node /opt/homebrew/lib/node_modules/@larksuite/cli/scripts/run.js ...
```

- Do not rely on `PATH` containing Homebrew binaries inside Codex threads; use
  absolute paths for both `node` and the CLI entrypoint.
- For wiki URLs, first resolve the wiki node to its `obj_token`, then operate
  on the underlying doc token.

## Useful Entry Points

- Main-agent entrypoint: `src/dataos/agent/loop.py`
- Agent CLI: `src/dataos/agent/cli.py`
- Runtime executors: `src/dataos/runtime`
- Artifact definitions: `src/dataos/core/artifact.py`
- Workspace definition: `src/dataos/core/workspace.py`
- PostTrainBench playbook: `resources/code/posttrainbench/PLAYBOOK.md`
- Main document: `docs/overview/dataos_overview.md`
