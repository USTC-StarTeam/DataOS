---
name: run-dataos-experiment
description: >-
  Drive a DataOS SFT data experiment end to end: load raw dataset(s) with the
  external dataos-operator library, compose operators (map to Alpaca/ShareGPT,
  filter, sample, mix) into one training set, export a training JSONL, then run
  the fixed stub train + eval + report stages via `dataos run-prepared`. YOU are
  the orchestrator/planner — there is no embedded planning agent. Use this
  whenever the user wants to build an SFT training set from data and run a DataOS
  experiment, prepare data for fine-tuning with DataOS, run a stub (no-GPU,
  no-LLM-key) DataOS pipeline, or mentions dataos-operator, run-prepared,
  Alpaca/ShareGPT data prep, or producing a DataOS report.json.
---

# Run a DataOS Experiment (skill-driven)

DataOS used to plan its own pipeline with an embedded LLM agent. In this skill
version, **control is inverted: you (the coding agent) are the planner.** You
decide how to turn raw data into a training set, you call the operators, and you
hand the finished training JSONL to a thin, deterministic runtime that only does
the fixed stages that must stay code (train + eval + finalize).

Two halves:

1. **Data stage (you drive, in Python).** Use the external `dataos-operator`
   library to load, transform, and export a training set. This is where all the
   judgement lives — which sources, which format, how much to keep.
2. **Run stage (one CLI call).** `dataos run-prepared` materializes your training
   JSONL as a dataset artifact and runs `run_train_stage` (stub) then
   `run_eval_stage` (provider=stub), then writes `report.json` and an experiment
   artifact. No GPU, no LLM API key, no planning agent.

Why split it this way: the operators and the train/eval primitives are
deterministic building blocks. The interesting, context-dependent part is
*composing* them, and that is exactly what you are good at. Keep the data stage
flexible; keep the run stage boring and reproducible.

## Follow This Order

1. **Understand the goal and locate inputs.** What training set is the user
   after, and from which source(s)? Inputs can be local `.jsonl`/`.json`/`.csv`,
   a Hugging Face dataset, or in-memory records. If you just need to prove the
   pipeline runs, a tiny synthetic dataset via `Dataset.from_records([...])` is
   the fastest path.

2. **Load each input with `dataos_operator.Dataset`.**
   - `Dataset.from_records([{...}, ...])` — in-memory rows.
   - `Dataset.from_local("path.jsonl")` — local jsonl/json/csv.
   - `Dataset.from_hf("openai/gsm8k", name="main", split="train[:100]", field_map={...})`
     — Hugging Face (needs the `hf` extra; not installed in stub-only setups).
   - `Dataset.from_source("hf://...")` / `Dataset.from_source("path")` — auto-routes.
   Note on bundled data: `datasets/gsm8k/` in this repo is a Hugging Face
   `DatasetDict` saved with `save_to_disk` (splits `train`/`test`, fields
   `question`/`answer`/`conversations`) — it is NOT a flat jsonl. `dataos_operator.Dataset`
   reads jsonl/json/csv files and HF hub paths, not `save_to_disk` directories, so
   export the rows to jsonl first (or just use `from_records`) before operating on them.

3. **Normalize toward a common shape, then mix.** Different sources have
   different fields. A reliable pattern is to map everything to OpenAI messages
   first with `OpenAIMessageMapper` (it understands `qa`, `alpaca`, `sharegpt`,
   `messages`), then merge with `ConcatMixer().run([ds_a, ds_b, ...])`. Mixing on
   a single shared schema avoids field-mismatch surprises.

4. **Filter and sample to taste.** Selectors live under
   `dataos_operator.operators.selector` — e.g. `LengthFilter(text_key=..., min_length=..., max_length=...)`
   and `RandomSampleSelector(sample_size=N, strict=False)`. Run
   `from dataos_operator.core import list_operator_specs` to see the full catalog
   (name, kind, description, class path) if you need clustering, dedup, scoring,
   etc.

5. **Map to the final training schema.** For Alpaca use
   `AlpacaMapper(source_format="messages")` → `instruction` / `input` / `output`.
   For ShareGPT use `ShareGPTMapper(source_format="messages")` → `conversations`
   (`[{"from": "human"/"gpt", "value": ...}]`). DataOS's train stage auto-detects
   the format: a `conversations` column ⇒ ShareGPT, otherwise Alpaca.

6. **Export the training JSONL.** `Dataset.save_local("train.jsonl")`. Keep only
   the training fields the trainer expects. For Alpaca that is `instruction`,
   `output`, and optionally `input`/`system`/`history`; operators may also carry
   a `metadata` field, which is harmless but you can drop it for a clean file.

7. **Run the fixed stages with `dataos run-prepared`.** Point it at a TaskTemplate
   JSON whose `train_config.mode` is `"stub"` and whose `eval_config.provider`
   is `"stub"`. The bundled template `dataos/templates/alpaca_gsm8k_stub_v1.json`
   already satisfies both. Command:
   ```bash
   dataos run-prepared \
     --task dataos/templates/alpaca_gsm8k_stub_v1.json \
     --dataset train.jsonl \
     --workspace workspace
   ```

8. **Confirm the report.** The command prints a JSON summary; the full report is
   at `<experiment_root>/report.json`. Check `status == "completed"`, that
   `steps` contains a completed `train_llamafactory` and `eval_model`, and that
   `artifacts` includes a `dataset`, a `model`, and an `eval`. The headline
   `score` comes from the stub eval's `provider_config.score` (0.0 by default;
   raise it in the template to simulate a higher score).

## Worked Example

Data stage (`prepare.py`) — synthetic sources, mapped + merged + sampled, then
exported as Alpaca JSONL:

```python
import random

from dataos_operator import Dataset
from dataos_operator.operators.mapper.alpaca_mapper import AlpacaMapper
from dataos_operator.operators.mapper.openai_message_mapper import OpenAIMessageMapper
from dataos_operator.operators.mixer.concat_mixer import ConcatMixer
from dataos_operator.operators.selector.random_sample_selector import RandomSampleSelector

# 1) Load input dataset(s).
qa = Dataset.from_records([
    {"question": "What is 2 + 3?", "answer": "5"},
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "What color is the sky on a clear day?", "answer": "Blue"},
    {"question": "How many legs does a spider have?", "answer": "Eight"},
])
alp = Dataset.from_records([
    {"instruction": "Reverse the word 'cat'.", "input": "", "output": "tac"},
    {"instruction": "Say hello in Spanish.", "input": "", "output": "Hola"},
])

# 2) Map both sources to a shared OpenAI-messages shape, then merge.
qa_msgs = OpenAIMessageMapper(source_format="qa", query_key="question", answer_key="answer").run(qa)
alp_msgs = OpenAIMessageMapper(source_format="alpaca").run(alp)
merged = ConcatMixer().run([qa_msgs, alp_msgs])

# 3) Sample, then map to the Alpaca training schema.
random.seed(7)
sampled = RandomSampleSelector(sample_size=6, strict=False).run(merged)
alpaca = AlpacaMapper(source_format="messages").run(sampled)

# 4) Export only the training fields as JSONL.
records = []
for r in alpaca:
    rec = {"instruction": r["instruction"], "output": r["output"]}
    if r.get("input"):
        rec["input"] = r["input"]
    records.append(rec)
Dataset.from_records(records).save_local("train.jsonl")
print(f"wrote {len(records)} records to train.jsonl")
```

Run stage — exact command:

```bash
python prepare.py
dataos run-prepared \
  --task dataos/templates/alpaca_gsm8k_stub_v1.json \
  --dataset train.jsonl \
  --workspace workspace
```

Expected printed summary (ids vary per run):

```json
{
  "task_id": "alpaca-gsm8k-stub-v1",
  "status": "completed",
  "score": 0.0,
  "num_steps": 2,
  "dataset_artifact_id": "dataset-...",
  "model_artifact_ids": ["model-..."],
  "eval_artifact_ids": ["eval-..."],
  "experiment_artifact_id": "exp-...",
  "report_path": "workspace/exp-.../report.json"
}
```

In `report.json`: `status` is `"completed"`, `steps` lists a completed
`train_llamafactory` (runtime) producing the model artifact and a completed
`eval_model` (runtime) producing the eval artifact, `artifacts` has the
dataset → model → eval lineage, and `plan`/`review` are `null` (no agent ran).

For ShareGPT instead of Alpaca, swap step 3's final map for
`ShareGPTMapper(source_format="messages")` and export the `conversations` field;
the trainer auto-detects ShareGPT from that column.

## Boundaries

- **You are the planner. Do NOT invoke `LLMPlanningAgent`, `dataos.runtime.run_task`,
  or anything under `dataos/prompts/` or `dataos/llm/`.** This skill is the
  agent-less path: `dataos run-prepared` (i.e. `dataos.runtime.prepared.run_prepared`)
  must run with no LLM API key. If you find yourself needing an API key, you are on
  the wrong entrypoint.
- **Operators come from the external `dataos-operator` package only.** Import from
  `dataos_operator` (e.g. `dataos_operator.operators.mapper.alpaca_mapper`). Do not
  use DataOS's local `dataos/operators/` for the data stage.
- **Stub mode is the contract here: no GPU, no training framework, no eval harness.**
  Keep `train_config.mode = "stub"` and `eval_config.provider = "stub"`. Real
  `llamafactory`/`lm-eval`/`vllm` modes need heavy extras that are intentionally not
  installed; do not switch to them in this flow.
- **The run stage owns train/eval/finalize; the data stage owns everything before
  the training JSONL.** `run-prepared` does not merge or clean data — it materializes
  the single JSONL you hand it as the working dataset and runs the fixed stages.
  Get the data right before calling it.
- **Produce a real training JSONL on disk and verify the real `report.json`.** Do
  not fabricate the summary; run the command and read the file it points to.
- **Keep the exported file to known training fields.** Alpaca:
  `instruction`/`output` (+ optional `input`/`system`/`history`). ShareGPT:
  `conversations` (+ optional `system`/`tools`). Unexpected non-string values in
  required fields will make the train stage raise.
```
