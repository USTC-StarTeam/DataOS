# PostTrainBench 子代理执行手册（per-experiment sub-agent playbook）

You are an **execution sub-agent**. The main-agent already fixed the recipe (data operator
pipeline + SFT hyperparameters); your job is to **realize ONE experiment** and write
`report.json`. Do NOT change the experiment intent, parent model, or eval scorer. You may make
minimal mechanical compatibility fixes while running (operator API quirks, fixed-pool schema
bridging, harmless parameter translation) and must record them in `report.json`.

Three steps: **build data (operators) → train (H100 GPU) → sampled eval (official scorer)
→ write report.json**.

## ⚠️ Login-node rules (READ FIRST)

The H100 login node is SHARED and fragile (no GPU). Abusing it stalls it for everyone — so:
- On the login node you may ONLY: write/submit an `sbatch` script, poll `sacct`/`squeue` (sleep
  ≥30s between polls), and run BOUNDED internet steps (the judge; `llm_*` operator scoring on a
  SMALL sample). Nothing else heavy.
- Do NOT run Python on the login node that imports `dataos`, `dataos_operator`, `datasets`, or
  reads fixed-pool datasets. Do NOT inspect operator source on the login node. Put validation
  imports and fixed-pool reads inside the Slurm job and inspect the job log instead.
- ALL data processing (operators, reading the pool, `save_to_disk`), training, and eval
  generation MUST run INSIDE the Slurm GPU job (`sbatch`) — NEVER directly on the login node.
- NEVER run a recursive / whole-filesystem scan: no `grep -r` / `find` / `du` over `/data…`.
  Read only the specific files you need — their exact paths are in the RUNTIME CONFIG block
  injected by the main-agent.

## Runtime config

The main-agent injects a `RUNTIME CONFIG` JSON block into your task prompt. It is derived
from the single user run JSON plus runtime defaults, and is the authoritative source for host,
paths, conda envs, Slurm options, cache dirs, and gateway settings. Do not use
machine-specific defaults from this playbook.

Expected keys:
- `cluster`: `host`, `user`, `port`, `key`, `partition`, `qos`, `gres`,
  `nodelist`, `cpus_per_task`
- `paths`: `deploy_root`, `run_root`, `runs_base`, `fixed_pool`, `conda_sh`, `train_env`,
  `eval_env`, `local_report`, `local_state`
- `gateway`: `base_url`, `model`, `key_file`
- `model`: `base_model`
- `posttrainbench`: `bench`, `profile`, `eval_limit`

When writing remote scripts, derive shell variables from that block:

```bash
AGENTVER=<paths.deploy_root>
RUN_ROOT=<paths.run_root>
FIXED_POOL=<paths.fixed_pool>
```

GPU work goes through Slurm with the configured `partition` / `qos` / `gres`. The login node
has internet; compute nodes may not.

## 1. Build data — run the operator recipe (the main lever)

The recipe is `{sources:[...], recipe:[{op,params},...]}`. Run it through `dataos` on H100
(the pool lives there), then bridge to the SFT columns. In conda `train`, this repo uses a
`src` layout, so set `PYTHONPATH=$AGENTVER/src:$AGENTVER`:

```python
# Sketch only: inspect the installed dataos/dataos_operator APIs in the Slurm job log and adapt.
# If fixed-pool sources have incompatible raw schemas, first bridge each row into a minimal
# common text schema needed by the selected operators and by train_sft: prompt/target/text.
# Then run the intended operator pipeline as closely as the installed operator APIs allow.
```
- Operators come from `dataos_operator`; `dataos.catalog()` lists the 22 (read it). The operator
  pipeline is the EXPERIMENT — it filters / dedups / quality-scores / samples the raw pool rows;
  the bridge above is just the mechanical format step train_sft needs at the end.
- If an operator rejects a parameter or an API differs from the plan text, make the smallest
  equivalent compatibility fix that preserves the experiment intent, then record it in
  `report.json` under a compatibility note.
- You own the operator and dataset ports. Inspect installed operator signatures/APIs inside the
  Slurm job, normalize fixed-pool source rows into fields expected by the selected operators and
  by `train_sft.py`, and keep a short compatibility log in the report.
- Where to run the data step: **fold it into the SAME Slurm GPU job as training** (step 2) — the
  compute node has the pool AND a GPU. So GPU-heavy operators are fine: `ifd/perplexity/deita/instag`
  read a local model's logits (need GPU), `embedding_*`/`domain2vec` use a small local model. The
  CPU operators (mappers / `concat_mixer` / dedup / `length_filter` / `random_sample_selector` /
  `ranking/score/set_coverage`) run there too. `llm_task_relevance_evaluator` /
  `llm_analysis_evaluator` call an LLM API — compute nodes may have NO internet, so if the recipe
  uses one, run ONLY the bounded API scoring step on the **login node** using `gateway.base_url`,
  `gateway.model`, and `gateway.key_file`. Do not read the full pool, run ordinary operators, or
  `save_to_disk` the final training set on the login node. Persist only the scored intermediate
  needed by the compute job, then sbatch the remaining selection / `save_to_disk` / train work.
  The core data-centric idiom: an evaluator writes a score → a `score_selector` /
  `ranking_selector` keeps the best rows.
- `smoke` profile → cap the final dataset to **≤1000 rows** (`random_sample_selector` last).
- If `dataos_operator` (or an operator's dep) is missing in the env, install it, then continue.

## 2. Train — LoRA SFT from the Base, on H100 GPU (Slurm)

Use the kept tool `train_sft.py` (do not rewrite it). Submit as a Slurm job (GPU):

```bash
python $AGENTVER/resources/code/posttrainbench/harness/train_sft.py \
  --base-model <BASE> --data-dir $RUN_ROOT/data/sft \
  --output-dir $RUN_ROOT/adapter --merged-dir $RUN_ROOT/final_model \
  --finetuning-type <lora|full> \
  --max-len <MAX_LEN> --bs <BS> --grad-accum <GRAD_ACCUM> \
  --lr <LR> --epochs <EPOCHS> --lora-r <LORA_R> --lora-alpha <LORA_ALPHA>
```
`--finetuning-type` comes from the recipe's `train.finetuning_type` (the main-agent decides
LoRA vs full-parameter). `lora` → trains adapters then merges into `final_model/`; `full` →
full-parameter SFT saved directly as `final_model/` (no merge; `lora_*` ignored). Either way
the result is a merged `final_model/`. Wrap step 1 (data, can be same job/CPU) + step 2 in one
Slurm script using the configured partition/qos/gres. Poll robustly because `sacct` may print
leading spaces:

```bash
sacct -j <id> --format=State --noheader -n \
  | awk '{$1=$1; print $1}' \
  | grep -qE '^(COMPLETED|FAILED|CANCELLED|TIMEOUT)$'
```

Retry ssh on transient network errors. Write the non-LLM data steps into the same job
(compute node has the pool + GPU). Try to keep the GPU busy: use sensible dataloader workers,
batch size / grad accumulation choices compatible with the plan, and avoid CPU-only stalls before
GPU work. For eval, use vLLM concurrency/settings that fit one GPU rather than serializing
unnecessarily.

## 3. Eval — the FROZEN official scorer, sampled during search

In conda `lm-eval`, with the offline env above:
```bash
python $AGENTVER/resources/code/posttrainbench/lm_eval_bridge/scripts/run_posttrainbench_protocol_eval.py \
  --task <BENCH> --model-path $RUN_ROOT/final_model --model-name ptb_<BENCH> \
  --output-dir $RUN_ROOT/eval/<BENCH>_limit<EVAL_LIMIT> \
  --posttrainbench-eval-src $AGENTVER/resources/code/posttrainbench \
  --benchmarks-dir $AGENTVER/resources/data/posttrainbench_benchmarks \
  --inspect-pkgs $AGENTVER/resources/code/inspect_evals \
  --python "$(command -v python)" --limit <EVAL_LIMIT>
```
- Reads the score from `<eval_dir>/summary.json` → `score` (native percentage).
- The search loop uses sampled eval by default. Use the injected `EVAL_LIMIT`; do not run a full
  evaluation unless the user/main thread explicitly asks for final validation.
- Defaults: `smoke` → `EVAL_LIMIT=20`; `formal` → `200` unless the runtime config overrides it.
- **Judge benches** (`healthbench`, `arenahardwriting`): the GPU job only generates predictions
  (compute node has no internet). Then run the DETERMINISTIC judge driver from the local repo —
  do NOT hand-roll it:
  ```bash
  .venv/bin/python -m dataos.integrations.posttrainbench.judge --eval-dir <eval_dir> --bench <BENCH> --model-alias ptb_<BENCH>
  ```
  It ssh's to the H100 login node, converts predictions, runs the configured judge, and prints
  `{"score": <0-100|null>, "metrics": {...}}`. Put that score into report.json.

## 4. Write the report (the contract)

Write `<exp_dir>/report.json` LOCALLY (the path is given to you) — this is what the main-agent
reads to attach the score to the tree:
```json
{"id":"exp-...","name":"<name>","status":"completed",
 "score":<float native % or null>,
 "stages":{"train":{"status":"completed"},"eval":{"status":"completed"}},
 "dataset":{"rows":<int>},
 "model":{"path":"<RUN_ROOT>/final_model","base_model":"<BASE>"},
 "eval":{"score":<float>,"per_task":{"<BENCH>":<float>},"path":"<eval_dir>"}}
```
`status:"completed"` only if `score` is a real number; otherwise `status:"failed"` with an
`"error"` string. Never crash without writing a report — a failed report is still required.

**Artifact layout is DETERMINISTIC — put every product at exactly these paths** (the runtime
records this layout onto the tree node for provenance; do not invent other locations):
`$RUN_ROOT/data/sft` (SFT data) · `$RUN_ROOT/adapter` (LoRA) · `$RUN_ROOT/final_model` (merged
model) · `$RUN_ROOT/eval/<BENCH>_limit<EVAL_LIMIT>` (eval outputs: summary.json/metrics/
predictions/judge) · `$RUN_ROOT/logs` · `<exp_dir>/report.json` (local). The runtime stamps a
`"artifacts"` block into report.json automatically, so the tree always knows where things are.
```
