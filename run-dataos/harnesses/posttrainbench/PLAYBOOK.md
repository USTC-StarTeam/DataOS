# PostTrainBench 子代理执行手册（per-experiment sub-agent playbook）

You are an **execution sub-agent**. The main-agent already fixed the recipe (data operator
pipeline + SFT hyperparameters); your job is to **realize ONE experiment** and write
`report.json`. Do NOT change the recipe or the eval scorer — only decide *how* to run things
and self-heal failures.

Three steps: **build data (operators) → train (H100 GPU) → eval (official scorer) → write report.json**.

## ⚠️ Login-node rules (READ FIRST)

The H100 login node is SHARED and fragile (no GPU). Abusing it stalls it for everyone — so:
- On the login node you may ONLY: write/submit an `sbatch` script, poll `sacct`/`squeue` (sleep
  ≥30s between polls), and run BOUNDED internet steps (the judge; `llm_*` operator scoring on a
  SMALL sample). Nothing else heavy.
- ALL data processing (operators, reading the pool, `save_to_disk`), training, and eval
  generation MUST run INSIDE the Slurm GPU job (`sbatch`) — NEVER directly on the login node.
- NEVER run a recursive / whole-filesystem scan: no `grep -r` / `find` / `du` over `/data…`.
  Read only the specific files you need — their exact paths are in this playbook, so you never
  have to search for anything (keys, configs, models are all named below).

## Resources

- **H100 (hcgu)**: `ssh -i ~/.ssh/ghc_miyao -o IdentitiesOnly=yes hcgu@114.214.215.159`. GPU work goes through Slurm: `sbatch --qos=gpu_priv --gres=gpu:1 --partition=gpu`. The login node has internet; **compute nodes do NOT**.
- **Deploy on H100**: `AGENTVER=/data/hcgu/users/tjh/dataos_agentver`. Per-experiment scratch: put everything under `RUN_ROOT=/data/hcgu/users/tjh/dataos_agentver_runs/<project>/<study>/<exp_slug>`.
- **conda**: `source /data/software/miniconda3/etc/profile.d/conda.sh` then `conda activate /data/hcgu/.conda/envs/train` (torch+peft, data+train) or `.../envs/lm-eval` (vllm+inspect, eval). Non-interactive ssh: always `source` first.
- **Fixed pool**: `/data/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v1_full` (`registry.json` lists sources + `benchmark_tags`; each source at `<pool>/<source_id>/train` is a `save_to_disk` dir).
- **Base models** (use the `-Base`, never instruct): `/data/hcgu/hf_cache/hub/<Qwen3-1.7B-Base|Qwen3-4B-Base|SmolLM3-3B-Base|gemma-3-4b-pt>`.
- **Offline eval env** (compute nodes): `export HF_HOME=/data/hcgu/hf_cache TRANSFORMERS_CACHE=/data/hcgu/hf_cache HF_DATASETS_CACHE=$AGENTVER/run-dataos/eval/benchmarks/hf_datasets HF_MODULES_CACHE=$AGENTVER/run-dataos/eval/benchmarks/modules HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`.
- **Gateway for LLM-scoring operators** (`llm_task_relevance_evaluator` / `llm_analysis_evaluator`): OpenAI-compatible `base_url=https://api2.starteam.wang/v1`, `model=claude-haiku-4-5` for bulk. The key is in a 0600 file on H100 — read it INLINE into the scoring process only, never print it and never search the filesystem for it: `OPENAI_API_KEY="$(cat /data/hcgu/users/tjh/.gateway_key)" OPENAI_BASE_URL=https://api2.starteam.wang/v1 python …`. (env vars do NOT cross ssh, so always set them inline on the remote command.)

## 1. Build data — run the operator recipe (the main lever)

The recipe is `{sources:[...], recipe:[{op,params},...]}`. Run it through `dataos` on H100
(the pool lives there), then bridge to the SFT columns. In conda `train`, `PYTHONPATH=$AGENTVER`:

```python
import json, dataos
from datasets import Dataset
from dataos._stage import run_recipe
pool="/data/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v1_full"
ds = None
for sid in SOURCES:                         # load + concat the chosen pool sources
    d = dataos.load(f"{pool}/{sid}/train")
    ds = d if ds is None else ds.concat(d)  # (or use concat_mixer inside RECIPE)
ds = run_recipe(ds, RECIPE)                 # the operator pipeline = the data-centric lever
recs = ds.to_records() if hasattr(ds, "to_records") else list(ds)
# BRIDGE to train_sft's contract — pool rows carry instruction/input/output; map them.
rows=[]
for r in recs:
    instr=(r.get("instruction") or r.get("query") or "").strip()
    inp=(r.get("input") or "").strip()
    out=(r.get("output") or r.get("answer") or "").strip()
    if not instr or not out: continue
    prompt = instr if not inp else f"{instr}\n\n{inp}"
    rows.append({"prompt":prompt,"target":out,"text":f"{prompt}\n\n{out}"})
Dataset.from_list(rows).save_to_disk(f"{RUN_ROOT}/data/sft")   # train_sft reads {prompt,target,text}
```
- Operators come from `dataos_operator`; `dataos.catalog()` lists the 22 (read it). The operator
  pipeline is the EXPERIMENT — it filters / dedups / quality-scores / samples the raw pool rows;
  the bridge above is just the mechanical format step train_sft needs at the end.
- Where to run the data step: **fold it into the SAME Slurm GPU job as training** (step 2) — the
  compute node has the pool AND a GPU. So GPU-heavy operators are fine: `ifd/perplexity/deita/instag`
  read a local model's logits (need GPU), `embedding_*`/`domain2vec` use a small local model. The
  CPU operators (mappers / `concat_mixer` / dedup / `length_filter` / `random_sample_selector` /
  `ranking/score/set_coverage`) run there too. `llm_task_relevance_evaluator` /
  `llm_analysis_evaluator` call an LLM API — compute nodes have NO internet, so if the recipe uses
  one, run that scoring step on the **login node** (gateway `base_url=https://api2.starteam.wang/v1`,
  key in env `OPENAI_API_KEY`, `model=claude-haiku-4-5`), persist the scored dataset, then sbatch
  train. The core data-centric idiom: an evaluator writes a score → a `score_selector` /
  `ranking_selector` keeps the best rows.
- `smoke` profile → cap the final dataset to **≤1000 rows** (`random_sample_selector` last).
- If `dataos_operator` (or an operator's dep) is missing in the env, install it, then continue.

## 2. Train — LoRA SFT from the Base, on H100 GPU (Slurm)

Use the kept tool `train_sft.py` (do not rewrite it). Submit as a Slurm job (GPU):

```bash
python $AGENTVER/run-dataos/harnesses/posttrainbench/train_sft.py \
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
`sbatch --qos=gpu_priv` script; poll `sacct -j <id> --format=State`. Retry ssh on transient
network errors. Write the data step into the same job (compute node has the pool + GPU).

## 3. Eval — the FROZEN official scorer (do not modify)

In conda `lm-eval`, with the offline env above:
```bash
python $AGENTVER/run-dataos/eval/dataos_lmeval/scripts/run_posttrainbench_protocol_eval.py \
  --task <BENCH> --model-path $RUN_ROOT/final_model --model-name ptb_<BENCH> \
  --output-dir $RUN_ROOT/eval/<BENCH>_limit<EVAL_LIMIT> \
  --posttrainbench-eval-src $AGENTVER/run-dataos/eval/posttrainbench_src \
  --benchmarks-dir $AGENTVER/run-dataos/eval/benchmarks \
  --inspect-pkgs $AGENTVER/run-dataos/eval/inspect_pkgs \
  --python "$(command -v python)" --limit <EVAL_LIMIT>
```
- Reads the score from `<eval_dir>/summary.json` → `score` (native percentage).
- `smoke` → `EVAL_LIMIT=20`; `formal` → `200`.
- **Judge benches** (`healthbench`, `arenahardwriting`): the GPU job only generates predictions
  (compute node has no internet). Then run the DETERMINISTIC judge driver from the local repo —
  do NOT hand-roll it:
  ```bash
  .venv/bin/python -m dataos_auto.judge --eval-dir <eval_dir> --bench <BENCH> --model-alias ptb_<BENCH>
  ```
  It ssh's to the H100 login node, converts predictions, runs the vendored Bailian judge
  (`qwen3.5-plus`, keys at `/data/hcgu/users/tjh/.bailian_keys`), and prints
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
