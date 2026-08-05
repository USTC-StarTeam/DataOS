# Online operators — hook protocol v1

Online operators interact with the **live training loop**; offline operators run upstream and
produce static canonical data. This document is the normative contract. It is owned by
**dataos-train** and is **backend-neutral**: backends realize hook points in their own
extension layer, and payloads carry plain Python types only (no torch tensors, no
backend objects).

The reference implementation and registry live in `dataos_train.online` (torch-free). The
LLaMA-Factory realization lives in `dataos_train.backends.llamafactory.ext`.

## Hook points

Three hook points exist: `reweight`, `select`, `step`. (`collate` was removed — batch
collation is backend plumbing, not an operator seam.)

| Hook point | Frequency | Call shape | Purpose |
|------------|-----------|------------|---------|
| `reweight` | every batch | `__call__(batch: dict) -> list[float]` | per-sample weight factors, mean ≈ 1 |
| `select`   | epoch boundary | `observe(batch) -> None` + `decide(ctx) -> dict` | choose/reweight data for the **next** epoch |
| `step`     | every optimizer step | `__call__(info: dict) -> None` | observe only (logging/scoring side effects) |

## Operator shape

Ops are constructed by a **factory**:

```python
factory(hook: str | None = None, **params)
```

- `hook=None` selects the op's own default hook point.
- An **explicitly passed** hook point the op does not support must raise `ValueError`
  at instantiation time — misconfiguration fails before a GPU job starts.

Instances are duck-typed (no shared base class, no import coupling):

```python
name: str
hook_point: str
def setup(self, ctx) -> None: ...
```

## Payloads

### `reweight` — every batch

`__call__(batch: dict) -> list[float]`, one factor per sample, normalized to mean ≈ 1 so
the op reshapes the signal without rescaling the loss globally.

Batch keys (all plain Python types):

| Key | Type | Meaning |
|-----|------|---------|
| `"losses"`      | `list[float]`      | per-sample losses from the live model |
| `"row_indices"` | `list[int] | None` | canonical row ids (LF backend: wired via an indexed dataset/collator wrapper; `None` for rows the wrapper cannot tag) |
| `"domains"`     | `None`             | reserved for domain labels |
| `"step"`        | `int`              | global optimizer step |
| `"epoch"`       | `int`              | current epoch (int floor) |

### `step` — observe only

`__call__(info: dict) -> None` with `info = {"step": int, "metrics": ...}`. A step op
must not try to change the training signal.

### `select` — epoch-boundary decisions (wired in the LLaMA-Factory backend)

- `observe(batch) -> None` is called during the epoch to accumulate statistics; it
  receives the same per-sample-loss payload as `reweight` hooks, every batch.
- `decide(ctx) -> {"indices": list[int] | None, "domain_weights": dict | None}` is called
  at an **epoch boundary** (`ctx = {"epoch": int, "step": int, "n_rows": int | None}`); the
  decision takes effect from the **next** epoch.
- Wiring: the train sampler is wrapped in `MutableSubsetSampler` (torch-free,
  `online/select.py`); a non-`None` `indices` decision narrows the next epoch to those
  rows, `None` restores pass-through. The dataloader re-iterates the sampler each epoch,
  so no trainer state has to be rebuilt.
- Any `interval` parameter of a select op is counted in **epochs**, not steps.

### Zero-marginal-cost discipline

An online op should cost ~nothing per step (a `reweight` op reuses losses the trainer
already computed; it does not run extra forward passes). This discipline is a **policy of
the ops themselves — the protocol does not enforce it**. An op that needs expensive
scoring belongs at `select` frequency (epoch boundary), not `reweight` frequency.

## Model jobs (offline scoring)

A **model job** is the offline sibling of an online op: instead of hooking the training
loop, it scores a static dataset with a model (forward passes only, no training) so
offline operators can consume per-sample signals. It is the layer-2 execution slot of the
operator stack (model × data, once) and lives in `dataos_train.modeljob` — torch-free
orchestration; torch enters only inside `modeljob/score_entry.py` at job time.

Spec shape (`ModelJobSpec`, `extra="forbid"`, peer of `TrainSpec`; `interaction` is a
discriminated union on `kind`):

```python
{
  "name": "l2-score",
  "model": "/models/Qwen3-1.7B-Base",              # injected by the runtime, never the LLM
  "dataset": {"path": "/data/curated", "format": "alpaca"},   # canonical DatasetRef
  "interaction": {"kind": "loss", ...},                      # or {"kind": "gradient", ...}
  "resources": {...}, "env": {...}, "cluster": {...}
}
```

CLI mirrors the train lifecycle (`submit`/`status` are deferred — the built JobSpec can be
run directly by an external sbatch wrapper via `bash run_score.sh`):

```bash
dataos-train modeljob-validate modeljob.json
dataos-train modeljob-build    modeljob.json [--run-dir DIR]   # default: modeljobs/<name> next to the dataset
dataos-train modeljob-collect  <run_dir>                       # -> ScoreResult (writes score_result.json)
```

Shared protocol:

- `build` renders `run_score.sh` (cd + `python -m dataos_train.modeljob.score_entry
  --config modeljob.json`) and the spec as `modeljob.json` into `run_dir`. The job runs in
  the env resolved from `env.backends["modeljob"]` (typically the training env).
- Records are mapped to chat messages (alpaca: `instruction` + optional `input` is the
  user turn, `output` the assistant turn; messages/sharegpt mapped accordingly) and the
  **model's own** chat template is applied (bf16, sdpa, batched).
- Product: `<run_dir>/scores.jsonl`, one `{"index": i, "key": output_key, <field>: float}`
  line per record (`loss` for kind=loss, `score` for kind=gradient); a `null` value marks
  a record whose assistant segment truncated away (counted as `metrics.n_null` by
  `collect`, excluded from stats). `collect` fails on missing/corrupt scores, non-finite
  values, gapped or duplicate indices, inconsistent `key`, or a row-count mismatch.
- The driving operator merges scores back by `index` and writes the value to the
  `output_key` field of each record — that is what offline `.run()` reads (e.g.
  `LossSelector(loss_key="metadata.runtime.loss")`).

### `kind: "loss"` (default)

Per-token mean cross-entropy over the **final assistant segment only**.
`{"batch_size": 4, "max_len": 512, "output_key": "metadata.runtime.loss"}`.
Verified equal to HF's built-in loss within 0.001 (see docs/online_ops_validation.md).

### `kind: "gradient"` (LESS-style, arXiv 2402.04333)

Each record gets a low-dimensional feature: the completion-only CE gradient w.r.t. a
temporary **untrained** LoRA adapter (base model frozen), sign-normalized, then projected
to `projection_dim` by a sparse Rademacher (JL) matrix — O(n) per record, streamed in
fixed blocks. `{"projection_dim": 1024, "projection_seed": 42, "lora_r": 8, "max_len": 512,
"target": DatasetRef | None, "output_key": "metadata.less.score"}`.

- `projection_seed` is shared by ALL candidate and target records (one projection
  matrix) — a hard constraint: per-record seeds make cosine similarities incomparable.
- With `target` given, `score` is the cosine similarity to the mean normalized target
  feature; without it, the projected vector's L2 norm. **`target` (D_val) must be
  task-representative** (bench train split / few-shot), not a slice of the candidate
  pool — see pitfall #5 in docs/training_pitfalls.md and the LESS verdict in
  docs/online_ops_validation.md.

Backend-neutral rules above apply unchanged: the scorer depends only on the HF
tokenizer/model in `model`, never on a training backend.

## Built-in reference ops

| Op | Hook | Params | Effect |
|----|------|--------|--------|
| `loss_reweight` | `reweight` | `alpha=1.0`, `clip=10.0` | factor_i ∝ (loss_i/mean)^alpha, mean-normalized — up-weights hard samples each batch |
| `epoch_subset_select` | `select` | `fraction=0.5` | epoch 1 pass-through; from epoch 2 on, narrows to the first `fraction` of rows (also the select-path smoke op) |

Both live in `dataos_train/online/reference.py`, registered in-module. Real-run benefit
matrix (no op beat random at 1.7B+LoRA+1k scale): docs/online_ops_validation.md.

## Discovery: in-module registry + entry points

Built-in ops register in-module via `register_online_op(name, factory)` and are resolved
by `resolve_online_ops(specs)` (`factory(hook=spec.get("hook"), **params)`).

External packages (e.g. `dataos_operator`) publish ops through the entry-point group
`dataos_train.online_ops` — duck-typed, with no import coupling to dataos-train:

```toml
# external package pyproject.toml
[project.entry-points."dataos_train.online_ops"]
my_op = "dataos_operator.online:MyOp"
```

`MyOp` only needs the duck-typed shape above (factory-callable with `hook=`, `name`,
`hook_point`, `setup`, `__call__`). Resolution loads the entry point lazily, only when the
op is named in a spec.

## Backend-neutral rules

- Payloads contain **plain Python types only** — backends convert out of their own tensor
  world before calling hooks.
- A new SFT backend writes its own `ext/` in the shape of
  `backends/llamafactory/ext/`: a thin trainer patch + launch entrypoint that import the
  training stack lazily and feed `dataos_train.online`.
- **verl is out of scope**: its online seam is reward injection (`method.reward.path`,
  reward == eval), which is a different mechanism and not part of this protocol.

## Trainer-patch discipline (LLaMA-Factory)

The LF realization is a **single-point monkeypatch** ("挂法 A"):

- `ext/entry.py` rebinds exactly one symbol —
  `llamafactory.train.sft.workflow.CustomSeq2SeqTrainer` — to a subclass built by
  `ext/trainer.py`, then delegates to LF's `run_exp`. No forked LF code.
- **Startup assertions** run after hook resolution and before patching: the symbol must
  exist and its `compute_loss` signature must still take `inputs`. Assertion failures
  raise a clear "LF version drifted" error instead of silently no-oping the patch.
- The **LF version is pinned**. The upgrade checkpoint when bumping LF is exactly two
  things: the symbol path (`llamafactory.train.sft.workflow.CustomSeq2SeqTrainer`) and
  the overridden `compute_loss(model, inputs, ...)` signature in `ext/trainer.py`.

## DataFlex alternative — considered, not adopted

[OpenDCAI/DataFlex](https://github.com/OpenDCAI/DataFlex) implements step-level dynamic
data selection via a "pause → score → rebuild dataloader" loop inside the trainer. That
can achieve finer-grained selection than our epoch-boundary `select`, but the
pause/rebuild cycle restarts the dataloader mid-epoch, which is a poor fit for
pretraining-scale (PTB) scenarios where startup and reshuffling costs dominate. We keep
the simpler epoch-boundary `select` semantics and do not adopt DataFlex.

## Deployment (runtime-pinned env)

Online ops run inside the training job, so both `dataos_train` and the op-provider package
(e.g. `dataos_operator`) must be importable from the job's Python env, with the provider
**installed** (entry-point discovery reads dist metadata — a bare `PYTHONPATH` checkout is
not enough).

Current arrangement (temporary): install both into the runtime-pinned env (e.g. the
`sft_env` uv venv from DataOS-Runtime's `env.yaml`), without touching its dependency set:

```bash
{env}/bin/python -m pip install --no-deps -e /path/to/DataOS-train -e /path/to/DataOS-Operator
```

`--no-deps` keeps the frozen venv's pinned versions untouched; the runtime deps each package
actually needs at job time are already there (`pydantic` for dataos-train, `numpy` for
dataos_operator — both come with the torch stack). Verify after install:

```bash
{env}/bin/python -c "from dataos_train.online import resolve_online_ops; \
print([h.name for h in resolve_online_ops([{'op': 'loss_reweighter'}])])"
{env}/bin/dataos-train capabilities
```

If `import dataos_train` fails on a missing `pydantic`, install it explicitly (it is
inert for the training stack). Longer term, per the runtime env discipline ("frozen shared
venvs; heterogeneous deps get their own environment"), train/op should move to a dedicated
environment — tracked as follow-up, not blocking.
