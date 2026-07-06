# PostTrainBench 子代理执行手册（per-experiment sub-agent playbook）

You are an **execution sub-agent**. The main-agent already fixed the recipe (data operator
pipeline + SFT hyperparameters); your job is to **realize ONE experiment** and write
`report.json`. Do NOT change the experiment intent, parent model, or eval scorer. You may make
minimal mechanical compatibility fixes while running (operator API quirks, fixed-pool schema
bridging, harmless parameter translation) and must record them in `report.json`.

Three steps: **build data (operators) → train (H100 GPU) → sampled eval (official scorer)
→ write report.json**.

**BASELINE experiment** — if the plan's `train` is EMPTY (no `finetuning_type`), SKIP steps 1–2:
do not build data or train. Evaluate the **parent model directly** (`--model-path` = the parent
model path, which on the first round is the base model) with step 3, and write `report.json` with
that model as `model.path`. This records the base model's reference score.

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

## ⚠️ Output isolation — keep the shared HOME (~) clean

EVERYTHING you create on H100 MUST live under `$RUN_ROOT`, never in the shared login-node HOME (`~`). Stray files in `~` (`slurm-*.out`, `torchinductor_*`, framework IPC sockets) break the cluster owner's housekeeping — treat a clean HOME as a hard requirement.

1. **First thing after ssh to the login node**, before any sbatch/python:
   ```bash
   mkdir -p $RUN_ROOT/logs $RUN_ROOT/tmp $RUN_ROOT/.cache && cd $RUN_ROOT
   ```
2. **Every `sbatch` script MUST pin its cwd + logs** (else `slurm-%j.out` lands in `~`):
   ```bash
   #SBATCH --chdir=<RUN_ROOT>
   #SBATCH --output=<RUN_ROOT>/logs/slurm-%j.out
   #SBATCH --error=<RUN_ROOT>/logs/slurm-%j.err
   ```
3. **Inside every GPU job, redirect framework caches / temp / sockets** into `$RUN_ROOT`:
   ```bash
   export TMPDIR=$RUN_ROOT/tmp
   export TORCHINDUCTOR_CACHE_DIR=$RUN_ROOT/.cache/inductor
   export TRITON_CACHE_DIR=$RUN_ROOT/.cache/triton
   export TORCH_HOME=$RUN_ROOT/.cache/torch
   ```

`slurm-*.out`, `torchinductor_*`, and framework IPC sockets all follow the process CWD / default cache dirs — pin them or they litter `~`.

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

## 0. RECON mode (only when your task is the RECON prompt, not a training experiment)

If your task prompt is the RECON prompt (gather intel → write `recon.json`), you do NOT train.
Produce the three parts and write the file:
1. **eval_format_contract** — read the {BENCH} task + scorer under
   `$AGENTVER/resources/code/posttrainbench/lm_eval_bridge/`; record how it parses model output and
   the EXACT `expected_target_shape` a training answer must take (incl. any template/tag).
2. **baseline_diagnosis** — decide if the base model's low score is a FORMAT miss vs a CAPABILITY
   gap (read a few base generations if present, or reason from base + contract). Don't run a full
   GPU eval just for this — set cause "unknown" if you can't tell cheaply.
3. **source_profiles** — run the deterministic profiler over the WHOLE pool (cheap, ~minutes):
   ```bash
   ssh <login> '<eval_env>/bin/python \
     $AGENTVER/resources/code/posttrainbench/lm_eval_bridge/scripts/profile_source.py \
     --pool-root <fixed_pool> --all-sources --target <function-calling|math|general>'
   ```
   It prints a JSON array of per-source `{n, domain, output_format, target_fit, quality, usage_hint}`.
   Put it in verbatim as `source_profiles` — do not hand-edit its verdicts.

## 1. Build data — run the operator recipe (the main lever)

The recipe is `{sources:[...], recipe:[{op,params},...]}`. `sources` are REAL source_ids the
main-agent already verified exist in the pool; each lives under the fixed pool (its dir is
`<fixed_pool>/<source_id>`, `fixed_pool` from RUNTIME CONFIG). Run the recipe through `dataos` on
H100 (the pool lives there), then bridge to the SFT format. In the configured `train_env`, this
repo uses a `src` layout, so set `PYTHONPATH=$AGENTVER/src:$AGENTVER`.

**LOOK BEFORE YOU WRITE.** Inside the Slurm job, FIRST load each source's dir and print a few real
rows + their columns, AND read the operators' own docs — they are the single source of truth for
the operator API, NOT this PLAYBOOK. Run `python -c "import dataos_operator as o; print(o.guide())"`
once for the framework how-to (data model, offline vs online use, the online runtime-config
schema), then `print(o.usage('<op_name>'))` for EACH operator in the recipe to get its exact
constructor params, run() contract, and config example. Only THEN write the bridge + operator
pipeline, shaped to the schema you actually saw — do NOT guess the API or write blind.

```python
# After inspecting the real rows: bridge each source row into the minimal common text schema the
# selected operators need (keep the recipe's intent), run the operator pipeline, then emit the
# FINAL training format = LLaMA-Factory alpaca ({instruction,input,output}) — see step 2.
```
- The operator pipeline (from `dataos_operator`, >=0.4.0) is the EXPERIMENT — it filters / dedups /
  quality-scores / samples the raw pool rows; the bridge above is just the mechanical format step
  LLaMA-Factory needs at the end.
- ONLINE operators (mode=online, e.g. `loss_selector` / `doremi_mixer` / `loss_reweighter`) run
  DURING SFT, not in the offline `.run()` pipeline. Follow `guide()`'s online section for the
  export → `dataos_runtime_config.json` → `dataos_enable` steps; do NOT also run them offline.
- If an operator rejects a parameter or an API differs from the plan text, make the smallest
  equivalent compatibility fix that preserves the experiment intent, then record it in
  `report.json` under a compatibility note.
- You own the operator and dataset ports. Inspect installed operator signatures/APIs inside the
  Slurm job, normalize fixed-pool source rows into fields expected by the selected operators and
  into the LLaMA-Factory alpaca format, and keep a short compatibility log in the report.
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
- `smoke` profile → cap the final dataset to **≤1000 rows** (`random_sample_selector` last).
- If `dataos_operator` (or an operator's dep) is missing in the env, install it, then continue.
- **WHEN THE OPERATORS CAN'T DO IT, HAND-WRITE A SCRIPT — don't force-fit.** The 22 operators have
  NO "keep only valid target-format rows" filter and no "normalize output to the eval's exact shape"
  step. If the recipe/brief needs cleaning or format-alignment they can't express, write a small
  standalone Python prep script (run it in the Slurm job) that: keeps only rows whose output parses
  as the eval's target shape, drops negatives (e.g. "no access to functions" rows), and renders the
  target with the SAME template the evaluator parses. This is how a BFCL run reached **83** — a
  hand-written `prepare_bfcl_sft.py` read the clean `xlam_function_calling_60k` source, triple-
  filtered (tools-parseable ∧ calls-parseable ∧ call-name∈tools), converted params to the BFCL
  JSON-schema, and rendered with the evaluator's own qwen3 template + `<tool_call>` tag. The run
  that instead random-sampled the dirty glaive source through plain operators scored **4**. Record
  the script + its filter/format logic in report.json.

## 2. Train — SFT from the parent model on H100 GPU (Slurm), with LLaMA-Factory

You WRITE the training yourself with **LLaMA-Factory** (`llamafactory-cli`, present in the
`train_env` from RUNTIME CONFIG together with `dataos_operator` — one env does data + train).
The main-agent's `train` dict (finetuning_type, learning_rate, num_train_epochs,
per_device_train_batch_size, gradient_accumulation_steps, lora_rank, lora_alpha, cutoff_len, …)
maps almost 1:1 onto LLaMA-Factory YAML keys — forward them, and you MAY use any LLaMA-Factory
training key the plan asks for (`stage`, `lora_target`, `warmup_ratio`, …). This is a
DEMONSTRATION: adapt paths/keys to the actual plan and the installed LF (`llamafactory-cli train -h`).

**2a. Bridge the recipe output → LLaMA-Factory dataset (alpaca).** Write the kept rows as an
alpaca JSON plus a `dataset_info.json`, into the deterministic data dir `$RUN_ROOT/data/sft`:

```python
# inside the Slurm job. `rows` = operator-pipeline output (each row: a question + its answer).
import json, os
d = f"{RUN_ROOT}/data/sft"; os.makedirs(d, exist_ok=True)
json.dump([{"instruction": r["prompt"], "input": "", "output": r["target"]} for r in rows],
          open(f"{d}/train.json", "w"), ensure_ascii=False)
json.dump({"dataos_sft": {"file_name": "train.json"}}, open(f"{d}/dataset_info.json", "w"))
# (alpaca columns instruction/input/output are LF's default; loss is on `output` only.)
```

**2b. Write the training YAML** (`$RUN_ROOT/train.yaml`) — example, fill from the plan's `train`:

```yaml
model_name_or_path: <PARENT_MODEL>     # RUNTIME CONFIG model.parent_model — never hardcode
stage: sft
do_train: true
finetuning_type: lora                  # or "full" — the plan decides; lora_* ignored for full
lora_target: all                       # lora only
dataset: dataos_sft
dataset_dir: <RUN_ROOT>/data/sft
template: qwen3                         # Qwen3 family; see template note below
cutoff_len: 1024
learning_rate: 1.5e-4
num_train_epochs: 1.0
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
lora_rank: 64
lora_alpha: 128
bf16: true
output_dir: <RUN_ROOT>/adapter         # lora → adapter dir; for full set this to <RUN_ROOT>/final_model
overwrite_output_dir: true
report_to: none
```

**2c. Train, then (lora) merge into the deterministic `final_model`:**

```bash
llamafactory-cli train $RUN_ROOT/train.yaml
# lora → merge adapter + base into final_model (full → output_dir already IS final_model):
llamafactory-cli export \
  --model_name_or_path <PARENT_MODEL> --adapter_name_or_path $RUN_ROOT/adapter \
  --template qwen3 --finetuning_type lora \
  --export_dir $RUN_ROOT/final_model --export_size 5
```
Either way `$RUN_ROOT/final_model` MUST end up a COMPLETE HF model — downstream eval and the
child node's continued training load it as a plain model (no adapter).

**template note:** the parent is a Qwen3 **Base** model. `template: qwen3` teaches the Qwen3 chat
format (matching how eval prompts it); `qwen3_nothink` drops thinking blocks for direct-answer
benches like GSM8K. The template is a legitimate experiment axis — pick one consistent with eval
and keep it stable across a tree branch (`llamafactory-cli train -h` / the LF template list).

Fold 2a+2b+2c **and** step 1's data work into ONE Slurm script (compute node has pool + GPU),
using the configured partition/qos/gres. Poll robustly because `sacct` may print leading spaces:

```bash
sacct -j <id> --format=State --noheader -n \
  | awk '{$1=$1; print $1}' \
  | grep -qE '^(COMPLETED|FAILED|CANCELLED|TIMEOUT)$'
```

Retry ssh on transient network errors. Keep the GPU busy: sensible dataloader workers and
batch/grad-accum compatible with the plan; avoid CPU-only stalls before GPU work. For eval, use
vLLM concurrency/settings that fit one GPU rather than serializing unnecessarily.

## 3. Eval — the lm-evaluation-harness protocol, sampled during search

GPU Slurm job, conda `eval_env` (lm-eval + vllm). The benchmark datasets are vendored IN THE
REPO at `$AGENTVER/resources/data/posttrainbench_benchmarks` (deployed with the code; its
`hf_datasets/` is a ready HF datasets cache for the deterministic benches, `local/` holds the
bfcl + judge-bench jsonl) — point lm-eval at it and force OFFLINE. `POSTTRAINBENCH_BENCHMARKS_DIR`
makes the class tasks (bfcl, healthbench, arenahardwriting) read their local jsonl from `local/`
(bfcl thus bypasses the HF-cache load entirely):
```bash
BENCH_DIR=$AGENTVER/resources/data/posttrainbench_benchmarks
export POSTTRAINBENCH_BENCHMARKS_DIR=$BENCH_DIR \
  HF_DATASETS_CACHE=$BENCH_DIR/hf_datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python $AGENTVER/resources/code/posttrainbench/lm_eval_bridge/scripts/run_lmeval_eval.py \
  --task <BENCH> --model-path $RUN_ROOT/final_model --model-name ptb_<BENCH> \
  --output-dir $RUN_ROOT/eval/<BENCH>_limit<EVAL_LIMIT> --limit <EVAL_LIMIT>
```
- Reads the score from `<eval_dir>/summary.json` → `score` (native percentage 0-100).
- The runner is self-contained (its task package + chat templates live next to it); it serves the
  model with vLLM and computes the lm-eval metric in-job.
- Sampled by default: use the injected `EVAL_LIMIT` (`smoke` 20 / `formal` 200 unless overridden);
  no full eval unless the user/main thread asks for a final validation.
- **Deterministic benches** (`gsm8k`, `humaneval`, `bfcl`, `gpqamain`, `aime2025`): scored in-job — done.
- **Judge benches** (`healthbench`, `arenahardwriting`): the same runner GENERATES predictions only
  (compute node has no internet) → writes `<eval_dir>/predictions.jsonl` + `summary.json`
  (`score:null`, `needs_judge:true`). Then on the H100 LOGIN node (has internet) run the judge,
  exporting the Bailian key inline (never write it to disk / report):
  ```bash
  ssh <login> 'export OPENAI_API_KEY=$(cat <judge key file>) \
    OPENAI_BASE_URL=<gateway/judge base_url> POSTTRAINBENCH_JUDGE_MODEL=<judge model> \
    POSTTRAINBENCH_BENCHMARKS_DIR=$AGENTVER/resources/data/posttrainbench_benchmarks; \
    <eval_env>/bin/python $AGENTVER/resources/code/posttrainbench/lm_eval_bridge/scripts/run_lmeval_judge.py \
      --bench <BENCH> --predictions <eval_dir>/predictions.jsonl --output <eval_dir>/summary.json \
      --workers 32'
  ```
  (`POSTTRAINBENCH_BENCHMARKS_DIR` lets the arena judge load its baseline answers from
  `local/arena-hard-v2.0/model_answer/` — the judge config `base_url`/`model`/`key_file`/`workers`
  all come from the RUNTIME CONFIG `judge` block. `--workers` (default 32) is the judge's document
  concurrency — predictions are scored in parallel; without it the judge runs near-serial and a
  HealthBench round can take ~45 min instead of ~10.)
  It applies the (pre-main) judge scoring and patches `summary.json` with the final `score`
  (0-100). Put that into report.json.

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
