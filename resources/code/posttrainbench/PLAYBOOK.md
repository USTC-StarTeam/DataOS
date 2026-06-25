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
  `eval_env`, `benchmarks`, `local_report`, `local_state`
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
(the pool lives there), then bridge to the SFT format. In the configured `train_env`, this
repo uses a `src` layout, so set `PYTHONPATH=$AGENTVER/src:$AGENTVER`:

```python
# Sketch only: inspect the installed dataos/dataos_operator APIs in the Slurm job log and adapt.
# If fixed-pool sources have incompatible raw schemas, first bridge each row into a minimal
# common text schema the selected operators need; the FINAL training format is LLaMA-Factory
# alpaca ({instruction,input,output}) — see step 2. Run the operator pipeline as the APIs allow.
```
- Operators come from `dataos_operator`; `dataos.catalog()` lists the 22 (read it). The operator
  pipeline is the EXPERIMENT — it filters / dedups / quality-scores / samples the raw pool rows;
  the bridge above is just the mechanical format step LLaMA-Factory needs at the end.
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
  The core data-centric idiom: an evaluator writes a score → a `score_selector` /
  `ranking_selector` keeps the best rows.
- `smoke` profile → cap the final dataset to **≤1000 rows** (`random_sample_selector` last).
- If `dataos_operator` (or an operator's dep) is missing in the env, install it, then continue.

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

GPU Slurm job, conda `eval_env` (lm-eval + vllm). The benchmark datasets are vendored at
`paths.benchmarks` (its `hf_datasets/` is a ready HF datasets cache for the 5 deterministic
benches; `local/` holds the judge-bench data) — point lm-eval at it and force OFFLINE:
```bash
export HF_DATASETS_CACHE=<paths.benchmarks>/hf_datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
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
  ssh <login> 'export OPENAI_API_KEY=$(cat <gateway/bailian key file>) \
    OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
    POSTTRAINBENCH_JUDGE_MODEL=qwen3.5-plus; \
    <eval_env>/bin/python $AGENTVER/resources/code/posttrainbench/lm_eval_bridge/scripts/run_lmeval_judge.py \
      --bench <BENCH> --predictions <eval_dir>/predictions.jsonl --output <eval_dir>/summary.json'
  ```
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
