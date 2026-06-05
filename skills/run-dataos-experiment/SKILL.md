---
name: run-dataos-experiment
description: >-
  Drive a DataOS SFT data experiment end to end in Python: load raw dataset(s)
  with dataos.load / the external dataos-operator library, compose operators
  (map to Alpaca/ShareGPT, filter, sample, mix) into one training Dataset, then
  call the staged DataOS API (experiment() -> set_plan -> train -> eval ->
  finalize) to run the deterministic train + eval + report stages. YOU are the
  orchestrator/planner — there is no embedded planning agent and no one-shot
  run(). Use this whenever the user wants to build an SFT training set from data
  and run a DataOS experiment, prepare data for fine-tuning with DataOS, run a
  stub (no-GPU, no-LLM-key) DataOS pipeline, or mentions dataos-operator,
  dataos.experiment, Alpaca/ShareGPT data prep, or producing a DataOS
  report.json.
---

# Run a DataOS Experiment (skill-driven)

DataOS is a library of **deterministic executors**; **you (the coding agent) are
the planner and orchestrator.** There is no embedded LLM, no planning agent, and
no `run()` that chains everything. You decide how to turn raw data into a
training set, you call the operators, and you call the five stages imperatively.

The five stages (call them yourself, in order, in Python):

1. **plan** — `exp.set_plan({...})` records the plan you decided on (no execution).
2. **data** — load with `dataos.load(...)`, then transform with `dataos_operator`
   operators. This is where all the judgement lives: which sources, which format,
   how much to keep.
3. **train** — `exp.train(dataset, config=...)` → a `Model`. Open config: any
   LLaMA-Factory key you pass in `config` is forwarded verbatim.
4. **eval** — `exp.eval(model, config=...)` → an `Eval`.
5. **review** — `exp.finalize()` writes `report.json` and returns it as a dict;
   YOU read it and decide what to do next.

Everything runs through one `Experiment` object. No CLI is required for the
pipeline (the only CLI subcommand is `dataos catalog`, a discovery helper).

## Follow This Order

1. **Understand the goal and locate inputs.** What training set is the user
   after, and from which source(s)? If you just need to prove the pipeline runs,
   a tiny synthetic dataset via `Dataset.from_records([...])` is the fastest path.

2. **Load each input.** `dataos.load(source, *, split="train", field_map=None,
   limit=None)` returns a `dataos_operator.Dataset` and routes by source:
   - `dataos.load("hf://openai/gsm8k", split="train", limit=100)` — Hugging Face hub.
   - `dataos.load("data/train.jsonl")` — local `.jsonl`/`.json`/`.csv` file.
   - `dataos.load("datasets/gsm8k", split="train", limit=8)` — a `save_to_disk`
     directory. The bundled `datasets/gsm8k/` is a HF `DatasetDict` saved with
     `save_to_disk` (splits `train`/`test`, fields `question`/`answer`/
     `conversations`); `dataos.load` reads it directly and selects the split.
   You can also call `dataos_operator.Dataset.from_records([...])` for in-memory rows.

3. **Discover operators when you need them.** `dataos.catalog()` returns a list of
   operator descriptors — `name`, `kind`, `description`, `operator_class`, and a
   `params` list (constructor parameter names, including conventional field keys
   like `query_key`/`answer_key`/`text_key`). Use it to find clustering, dedup,
   scoring, etc. (`dataos catalog` prints the same thing from the shell.)

4. **Normalize toward a common shape, then mix.** A reliable pattern is to map
   every source to OpenAI messages first with `OpenAIMessageMapper` (understands
   `qa`, `alpaca`, `sharegpt`, `messages`), then merge with
   `ConcatMixer().run([ds_a, ds_b, ...])`. Mixing on one shared schema avoids
   field-mismatch surprises.

5. **Filter and sample to taste.** Selectors live under
   `dataos_operator.operators.selector` — e.g.
   `LengthFilter(text_key=..., min_length=..., max_length=...)` and
   `RandomSampleSelector(sample_size=N, strict=False)`. Each operator runs as
   `Op(**params).run(dataset) -> Dataset`.

6. **Map to the final training schema.** For Alpaca use
   `AlpacaMapper(source_format="messages")` → `instruction` / `input` / `output`.
   For ShareGPT use `ShareGPTMapper(source_format="messages")` → `conversations`
   (`[{"from": "human"/"gpt", "value": ...}]`). The train stage auto-detects the
   format: a `conversations` column ⇒ ShareGPT, otherwise Alpaca.

7. **Start the experiment and record your plan.**
   `exp = dataos.experiment()` (or `dataos.experiment(task=<TaskTemplate-or-dict>,
   workspace="workspace")`). With no `task`, a built-in stub task is used. Then
   `exp.set_plan({"steps": [...]})`.

8. **Train.** Pass the training `Dataset` straight to `exp.train`:
   - Stub (no GPU / no extras / no LLM): `exp.train(ds, config={"mode": "stub"})`.
   - Real LLaMA-Factory: `exp.train(ds, config={"model_name_or_path": ...,
     "template": ..., "finetuning_type": "lora", ...})`. The config is **open** —
     every non-reserved key is forwarded to the generated `train.yaml`. DataOS owns
     and rejects the reserved keys `stage`, `do_train`, `dataset`, `eval_dataset`,
     `dataset_dir`, `output_dir`, `overwrite_output_dir`; `model_name_or_path` and
     `template` are required and `finetuning_type` must be `lora`/`full`/`freeze`.
     (Real mode needs the `train` extra + a GPU; stub mode needs neither.)

9. **Eval.** `exp.eval(model, config=...)`:
   - Stub: `exp.eval(m, config={"provider": "stub", "provider_config": {"score": 0.0}})`.
   - Real lm-eval (needs the `eval` extra): e.g.
     `exp.eval(m, config={"mode": "lm_eval_native", "model": "hf",
     "tasks": ["gsm8k"], "limit": 50, "num_fewshot": 0, "apply_chat_template": True})`.

10. **Finalize and confirm.** `rep = exp.finalize()` writes
    `<workspace>/<exp_id>/report.json` and returns it. Check `rep["status"] ==
    "completed"`, that `rep["steps"]` contains completed `train` and `eval` steps,
    and that `rep["dataset"]` / `rep["model"]` / `rep["eval"]` are populated. Read
    the file it wrote; do not fabricate it.

## Worked Example (stub, no GPU / no LLM key)

```python
import dataos
from dataos_operator.operators.mapper.alpaca_mapper import AlpacaMapper
from dataos_operator.operators.mapper.field_project_mapper import FieldProjectMapper
from dataos_operator.operators.mapper.openai_message_mapper import OpenAIMessageMapper

# 2) Data stage — load the bundled gsm8k save_to_disk dir and map to Alpaca.
ds = dataos.load("datasets/gsm8k", split="train", limit=8)
msgs = OpenAIMessageMapper(source_format="qa", query_key="question", answer_key="answer").run(ds)
alpaca = AlpacaMapper(source_format="messages").run(msgs)   # -> instruction / output columns
# Drop the source 'conversations' column so the train stage auto-detects Alpaca (not ShareGPT):
alpaca = FieldProjectMapper(keep_fields=("instruction", "input", "output"), strict=False).run(alpaca)

# 7) Start the experiment and record the plan.
exp = dataos.experiment()                                   # stub task; workspace="workspace"
exp.set_plan({"steps": ["load", "map", "train", "eval"]})

# 8-9) Train + eval in stub mode.
model = exp.train(alpaca, config={"mode": "stub"})
evaluation = exp.eval(model, config={"provider": "stub", "provider_config": {"score": 0.0}})

# 10) Review — write and read report.json.
report = exp.finalize()
assert report["status"] == "completed"
print(report["id"], report["model"]["id"], report["eval"]["score"])
```

`report.json` (flat) looks like:

```json
{
  "id": "exp-...",
  "status": "completed",
  "task": {"id": "stub", "name": "Stub Experiment", "goal": "..."},
  "plan": {"steps": ["load", "map", "train", "eval"]},
  "steps": [
    {"name": "create",   "status": "completed", "at": "..."},
    {"name": "plan",     "status": "completed", "at": "..."},
    {"name": "train",    "status": "completed", "at": "...", "detail": {"mode": "stub"}},
    {"name": "eval",     "status": "completed", "at": "...", "detail": {"score": 0.0}},
    {"name": "finalize", "status": "completed", "at": "..."}
  ],
  "dataset": {"path": "workspace/exp-.../dataset", "format": null, "rows": 8},
  "model":   {"id": "model-...", "path": "workspace/exp-.../model", "base_model": null, "finetuning_type": null, "mode": "stub"},
  "eval":    {"id": "eval-...", "score": 0.0, "metrics": {}, "path": "workspace/exp-.../eval"}
}
```

For ShareGPT instead of Alpaca, swap the final map for
`ShareGPTMapper(source_format="messages")` and keep the `conversations` column;
the trainer auto-detects ShareGPT from that column.

## Boundaries

- **You are the planner. There is no embedded agent and no LLM key.** DataOS no
  longer ships a planning agent, prompts, or an LLM client — the whole pipeline is
  driven by your Python calls. If you find yourself reaching for an API key to run
  the pipeline, you are doing it wrong.
- **No one-shot `run()`.** Call the stages yourself: `set_plan` → `train` → `eval`
  → `finalize`. Each is a separate, explicit call on the `Experiment`.
- **Operators come from the external `dataos-operator` package only.** Import from
  `dataos_operator` (e.g. `dataos_operator.operators.mapper.alpaca_mapper`), or
  discover them via `dataos.catalog()`. DataOS does not ship its own operators.
- **Datasets are `dataos_operator.Dataset`, not a DataOS type.** `dataos.load`
  returns one; operators consume and return one; `exp.train` takes one. Do not
  wrap it in any artifact type.
- **Train/eval are open-config executors.** Pass any LLaMA-Factory / lm-eval knob
  through the `config` dict — there is no allowlist. DataOS only reserves the few
  keys it must own (listed in step 8) and validates the required ones.
- **Stub mode is the no-GPU contract.** `config={"mode": "stub"}` (train) and
  `config={"provider": "stub", ...}` (eval) run with only `datasets` +
  `dataos-operator` installed — no torch, no lm-eval, no vllm. Real
  `llamafactory` / `lm_eval_native` modes need the `train` / `eval` extras and a
  GPU; only switch to them when those are available.
- **Verify the real `report.json`.** Run the pipeline and read
  `<workspace>/<exp_id>/report.json`; do not fabricate the summary.
- **Keep training fields to known shapes.** Alpaca: `instruction`/`output`
  (+ optional `input`/`system`/`history`). ShareGPT: `conversations`
  (+ optional `system`/`tools`). Unexpected non-string values in required fields
  make the train stage raise.
```
