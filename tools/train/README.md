# DataOS-train

A unified, multi-backend **training layer** for [DataOS](../..) (the main monorepo — this tool is
its `tools/train` workspace member). An agent emits a
declarative `TrainSpec` (facts only) and gets back a `TrainResult`; the consequential steps
between — config materialization, data conversion, metric extraction, checkpoint
normalization — are deterministic code, not LLM improvisation.

The design separates three **orthogonal axes**:

| Axis | What it picks | This release |
|------|---------------|--------------|
| **Backend** | which training library | `llamafactory` (SFT), `verl` (GRPO/RL), `dummy` (tests) |
| **Launcher** | where/how a job runs | `slurm` (sbatch + sacct), `local` (bare-metal subprocess) |
| **Data** | the canonical record schema | alpaca / messages / sharegpt (SFT), prompt+ground_truth (RL) |

The agent's surface stays tiny: it never writes a `train.yaml` or a verl config, never greps a
log — it picks a backend and method and reads one result.

## Install

```bash
pip install -e ".[dev]"                  # core (pydantic) + pytest/ruff
pip install -e ".[llamafactory]"         # to run the LLaMA-Factory backend locally
# verl runs from its own conda env on the cluster (see "verl" below)
```

Python ≥ 3.11.

> Deployment note: despite the `conda` naming in this document, the cluster envs DataOS runs
> against are frozen uv venvs (declared in env.yaml `tools`); the `TrainSpec.env.conda_sh` field
> is kept for compatibility.

## The contract

```python
from dataos_train import TrainSpec, get_backend, get_launcher

spec = TrainSpec(
    name="exp-1", backend="llamafactory",
    base_model="/models/Qwen3-1.7B-Base",      # injected by DataOS; never written by the LLM
    run_dir="/runs/exp-1",
    dataset={"path": "/data/curated", "format": "messages"},
    method={"kind": "sft", "finetuning_type": "lora", "learning_rate": 1.5e-4},
    resources={"nodes": 1, "gpus": 1, "cpus": 8, "time_h": 6},
    env={"conda_sh": "/c/conda.sh", "backends": {"llamafactory": "Llama"}, "deploy_root": "/deploy"},
    cluster={"partition": "gpu", "qos": "gpu_priv", "gres": "gpu:1", "nodelist": "gpu04"},
)

backend = get_backend(spec.backend)
errors  = backend.validate(spec)            # fail-fast, no GPU
job     = backend.build(spec)               # canonical->native + rendered assets + command
handle  = get_launcher("slurm").submit(job) # sbatch; returns immediately
# ... poll get_launcher("slurm").status(handle) until terminal ...
result  = backend.collect(job.run_dir)      # -> TrainResult (model_path, metrics, report_fragment)
```

`TrainResult.report_fragment` is a ready-to-merge `{stages.train, model, dataset}` subtree
matching DataOS `report.json`.

## CLI (JSON in / JSON out) — the surface a sub-agent drives

```bash
dataos-train validate     trainspec.json
dataos-train build        trainspec.json            # -> JobSpec  (writes <run_dir>/spec.json)
dataos-train submit       spec.json --launcher slurm # -> Handle  (writes handle.json, returns now)
dataos-train status       handle.json                # -> {status, handle}
dataos-train logs         handle.json --tail 50
dataos-train collect      handle.json                # -> TrainResult (writes result.json)
dataos-train run          trainspec.json --launcher local   # blocking submit+poll+collect (local/CI)
dataos-train capabilities [backend]
dataos-train schema [name]                           # machine-readable TrainSpec/method schemas
dataos-train schema --backend llamafactory           # capabilities + method schemas for a backend
dataos-train schema --method sft                     # parameter schema for one method

# model jobs (score data with a model, no training — see "Model jobs" below)
dataos-train modeljob-validate modeljob.json
dataos-train modeljob-build    modeljob.json [--run-dir DIR]  # renders run_score.sh + modeljob.json
dataos-train modeljob-collect  <run_dir>                      # -> ScoreResult (writes score_result.json)
```

Multi-hour cluster jobs use `submit` + detached `status` polling (handle.json on disk is the
resume-safe source of truth); `run` is for local/CI/short jobs.

Upstream agents should use `capabilities` to discover available backends/methods, `schema` to
discover typed parameters/defaults, and `validate` as the pre-submit gate. Runtime-owned fields
such as `base_model`, `run_dir`, `dataset.path`, env activation, native train configs, and Slurm
scripts should be injected or rendered by DataOS/DataOS-train rather than improvised by the agent.

## Backends

- **llamafactory** (SFT) — converts canonical data to alpaca, renders the LF config, runs
  `llamafactory-cli train` and (for LoRA) merges to a complete HF `final_model`.
- **verl** (GRPO) — converts canonical RL data to verl parquet (prompts wrapped in the *same*
  template the evaluator uses), runs `verl.trainer.main_ppo`, and merges the checkpoint to HF
  `final_model` in-job. Its reward is an **injected file path** (`method.reward.path`, runtime-owned): the driving
  agent writes `compute_score` from the task's frozen format contract — *reward == eval* is
  the agent's discipline, not a library vendoring. Requires a separate `verl` conda env
  (`env.backends.verl`). RL prompts should arrive already eval-template-aligned (or pass
  `--template-file` to the data converter).

### Env domains

`env.backends` is keyed by **env domain** — the backend name by default, or a method-scoped
key when a method's dependency set conflicts with the backend default. Each backend
advertises its non-default domains in `Capabilities.env_domains`; the method-scoped key
wins at resolution time. Today's only non-default domain is LLaMA-Factory PPO, whose TRL
pin conflicts with the standard LF env (verified on yzc-3090-1; see
`envs/llamafactory-ppo-cu121-py311.constraints.txt`):

```python
spec.env.backends = {
    "llamafactory": "/envs/dataos-train-lf-cu121-py311",          # SFT / DPO / ORPO / KTO
    "llamafactory:ppo": "/envs/dataos-train-lf-ppo-cu121-py311",  # PPO
}
```

`validate` rejects a PPO spec that only provides the generic `llamafactory` env, so the
mismatch is caught before a GPU job is submitted rather than at runtime.

## Online operators (data ↔ training interaction)

Offline operators run upstream and produce static canonical data. **Online operators** interact
with the live training loop and are a first-class seam, defined by the backend-neutral
**hook protocol v1** (`reweight` / `select` / `step`) in [docs/online_ops.md](docs/online_ops.md).
The protocol, registry, and reference ops live in `dataos_train.online` (torch-free and
unit-tested); LLaMA-Factory realizes all three hook points via a contained extension
(`backends/llamafactory/ext/`) that subclasses the LF trainer (per-sample losses fed from a
`compute_loss` patch; `select` decisions applied through a `MutableSubsetSampler` at epoch
boundaries; canonical `row_indices` wired through an indexed dataset/collator wrapper).
Two reference ops ship in-module:

| Op | Hook | Params | Effect |
|----|------|--------|--------|
| `loss_reweight` | `reweight` | `alpha=1.0, clip=10.0` | up-weights hard (high-loss) samples each batch, mean ≈ 1 |
| `epoch_subset_select` | `select` | `fraction=0.5` | epoch 1 full pass; later epochs narrow to the first `fraction` of rows |

```python
spec.online_ops = [{"op": "loss_reweight", "hook": "reweight", "params": {"alpha": 1.0}}]
```

External packages (e.g. `dataos_operator`) publish ops through the `dataos_train.online_ops`
entry-point group — see [docs/online_ops.md](docs/online_ops.md). Real-run validation of the
whole seam (all three hook points, row_indices, entry-point discovery) plus the benefit
matrix: [docs/online_ops_validation.md](docs/online_ops_validation.md). Battle-tested
training pitfalls (cutoff/termination, untrained special tokens, scorer format contracts,
D_val discipline, HF Trainer batch-key traps): [docs/training_pitfalls.md](docs/training_pitfalls.md).

## Model jobs (score data with a model, no training)

A **model job** is the offline sibling of an online op — the layer-2 execution slot of the
operator stack (model × data, one forward pass): it scores a static dataset so downstream
offline operators can consume per-sample signals. `ModelJobSpec` is a peer of `TrainSpec`
(same validate/build/collect lifecycle, same reused blocks), emitted by
`dataos_train.modeljob` (torch-free orchestration; torch enters only inside the job).
Two interaction kinds:

- `loss` — per-sample completion-only cross-entropy → `scores.jsonl` (`{"index", "key", "loss"}`).
- `gradient` — LESS-style features (arXiv 2402.04333): completion-only CE gradient w.r.t. a
  temporary untrained LoRA adapter, sign-normalized, projected by a shared-seed sparse
  Rademacher matrix; with a `target` dataset the score is cosine similarity to the mean
  target feature (D_val must be task-representative — see pitfalls #5), else the L2 norm.

Full contract: [docs/online_ops.md](docs/online_ops.md) §Model jobs.

## Tests

```bash
pytest -q          # default: dummy+local orchestration, slurm render, online hooks, backend build/collect
ruff check src tests
```
Tests needing a scheduler/GPU are marked `needs_slurm` / `needs_gpu` and skipped by default.

## Status & follow-ups

- Backends + launchers are unit/render-verified end to end; real verl GPU runs are gated on
  provisioning a `verl` conda env on the cluster.
- DataOS integration is wired and battle-tested: DataOS's Trainer registry entry `DataosTrainCli`
  drives this CLI's validate/build/submit/status/collect lifecycle (see the main repo's AGENTS.md).
- Deferred: a wider online-operator catalog (the three hook points are wired and validated;
  at 1.7B+LoRA+1k scale no shipped selection method beat random sampling — see
  `docs/online_ops_validation.md`), verl LoRA + multi-node, a TRL backend for CI.
