---
name: run-dataos-experiment
description: >-
  Drive DataOS SFT experiments end to end in Python: load raw dataset(s) with
  dataos.load / the external dataos-operator library, compose operators into
  training Datasets, then run one or many experiments under a dataos.study(...)
  container — study.set_plan(plan) -> study.experiment(spec) -> train -> eval ->
  finalize per experiment -> study.compare() -> study.finalize(conclusion). YOU
  are the runtime/planner: you write the multi-experiment plan, drive the loop,
  and decide whether to parallelize (DataOS does not). Use this whenever the user
  wants to build SFT training sets and run DataOS experiments, sweep/compare
  several training configs, prepare data for fine-tuning with DataOS, run a stub
  (no-GPU) DataOS pipeline, or mentions dataos-operator, dataos.study,
  dataos.experiment, Alpaca/ShareGPT data prep, or a DataOS report/review.
---

# Run DataOS Experiments (skill-driven)

DataOS is a library of **deterministic executors** plus two containers —
**`Study`** (a set of experiments sharing one plan) and **`Experiment`** (one
run). **You (the coding agent) are the runtime and planner.** There is no
embedded LLM, no planning agent, no `run()` that chains stages, and **no
scheduler** — whether to run experiments in parallel is your call, not DataOS's.

A plan usually describes **more than one experiment** (sweep a learning rate,
compare LoRA vs full, try data recipes). The shape:

```
plan = {goal, experiments: [spec, ...]}            # you write this
study = dataos.study(id); study.set_plan(plan)     # container (assigns a 编号)
  for spec in plan["experiments"]:
    exp = study.experiment(spec) -> data -> train -> eval -> finalize   # per-exp report.json
study.compare()  ->  study.finalize(conclusion)    # deterministic table -> review.json
```

## Stages (call them yourself, in Python)

1. **plan** — write a `plan` dict: `{"goal": ..., "experiments": [spec, ...]}`.
   Each `spec` is `{"name", "data", "train", "eval"}`. Use `dataos.catalog()` to
   discover operators while planning. DataOS does **not** generate the plan.
2. **study** — `study = dataos.study(id="my-sweep")` creates (or **reopens**) the
   container; omit `id` to auto-assign a serial `study-0001`. `study.set_plan(plan)`
   records it and writes `plan.json`.
3. **data** — per experiment, `dataos.load(...)` then transform with
   `dataos_operator` operators. All the judgement lives here. A processed
   `Dataset` is a plain value — **reuse it across experiments** if they share data.
4. **train** — `exp.train(dataset, config=None)` → `Model`. `config` defaults to
   `spec["train"]`. Open config: any LLaMA-Factory key is forwarded verbatim.
5. **eval** — `exp.eval(model=None, config=None)` → `Eval`. `model` defaults to the
   experiment's own trained model; `config` defaults to `spec["eval"]`.
6. **finalize (per experiment)** — `exp.finalize()` writes that experiment's
   `report.json` (its summary).
7. **review** — `study.compare()` reads every experiment's `report.json` off disk
   and returns a deterministic comparison (varied config dims + ranking + best).
   YOU read it and write the conclusion; `study.finalize(conclusion=...)` writes
   `review.json`. DataOS tabulates; you conclude.

One experiment, no sweep? Skip the study: `dataos.experiment(spec=...)` returns a
standalone `Experiment` (flat `workspace/<exp_id>/`).

## Follow This Order

1. **Understand the goal and design the set of experiments.** What is the user
   after, and **what varies across experiments** (the swept dimension)? What data
   does each need? If you just need to prove the pipeline runs, a tiny synthetic
   set via `Dataset.from_records([...])` and stub mode is the fastest path.

2. **Load each input.** `dataos.load(source, *, split="train", field_map=None,
   limit=None)` returns a `dataos_operator.Dataset` and routes by source:
   - `dataos.load("hf://openai/gsm8k", split="train", limit=100)` — Hugging Face hub.
   - `dataos.load("data/train.jsonl")` — local `.jsonl`/`.json`/`.csv` file.
   - `dataos.load("datasets/gsm8k", split="train", limit=8)` — a `save_to_disk`
     directory (HF `DatasetDict`); it selects the split for you.
   `dataos_operator.Dataset.from_records([...])` builds one from in-memory rows.

3. **Discover operators when you need them.** `dataos.catalog()` returns operator
   descriptors — `name`, `kind`, `description`, `operator_class`, and a `params`
   list (constructor params, including field keys like `query_key`/`answer_key`).
   (`dataos catalog` prints the same from the shell — the only CLI subcommand.)

4. **Normalize toward a common shape, then mix.** Map every source to OpenAI
   messages first with `OpenAIMessageMapper` (understands `qa`, `alpaca`,
   `sharegpt`, `messages`), then merge with `ConcatMixer().run([ds_a, ds_b, ...])`.
   Mixing on one shared schema avoids field-mismatch surprises.

5. **Filter and sample to taste.** Selectors live under
   `dataos_operator.operators.selector` — e.g. `LengthFilter(...)` and
   `RandomSampleSelector(sample_size=N, strict=False)`. Each runs as
   `Op(**params).run(dataset) -> Dataset`.

6. **Map to the final training schema.** Alpaca:
   `AlpacaMapper(source_format="messages")` → `instruction`/`input`/`output`.
   ShareGPT: `ShareGPTMapper(source_format="messages")` → `conversations`. The
   train stage auto-detects: a `conversations` column ⇒ ShareGPT, else Alpaca. (If
   you mapped to Alpaca but a stray `conversations` column remains, drop it with
   `FieldProjectMapper(keep_fields=("instruction","input","output"), strict=False)`.)

7. **Open the study and record the plan.**
   `study = dataos.study(id="...")`, then `study.set_plan({"goal": ...,
   "experiments": [spec, ...]})`.

8. **Run each experiment.** Loop over `study.plan["experiments"]`:
   `exp = study.experiment(spec)`, build the `Dataset`, `exp.train(ds)`,
   `exp.eval()`, `exp.finalize()`. Train config (from `spec["train"]`) is **open** —
   every non-reserved key is forwarded to the generated `train.yaml`. DataOS owns
   and rejects the reserved keys `stage`, `do_train`, `dataset`, `eval_dataset`,
   `dataset_dir`, `output_dir`, `overwrite_output_dir`; `model_name_or_path` and
   `template` are required and `finetuning_type` must be `lora`/`full`/`freeze`.
   Give each experiment a **distinct, short `name`** — it becomes the directory
   slug and the join key `compare()` uses.

9. **Review.** `cmp = study.compare()` → deterministic comparison; read it, then
   `study.finalize(conclusion="...")` writes `review.json`. Read the files; do not
   fabricate them.

## Worked Example (stub, no GPU / no LLM key)

```python
import dataos
from dataos_operator import Dataset

def alpaca(n):  # tiny synthetic Alpaca set (real recipe goes in the data stage)
    return Dataset.from_records(
        [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(n)],
        metadata={"dataset_format": "alpaca"},
    )

# plan (two experiments) + study
study = dataos.study(id="lr-smoke")
study.set_plan({
    "goal": "compare two stub configs",
    "experiments": [
        {"name": "a", "train": {"mode": "stub", "learning_rate": 1e-4},
         "eval": {"provider": "stub", "provider_config": {"score": 0.1}}},
        {"name": "b", "train": {"mode": "stub", "learning_rate": 5e-5},
         "eval": {"provider": "stub", "provider_config": {"score": 0.2}}},
    ],
})

# run each experiment (train/eval config default from the spec)
for spec in study.plan["experiments"]:
    exp = study.experiment(spec)
    ds = alpaca(8)
    exp.train(ds)        # -> spec["train"]
    exp.eval()           # -> spec["eval"], evaluates exp.model
    exp.finalize()       # -> experiments/<name>/report.json

# review
cmp = study.compare()                            # {varied_keys, ranking, best, ...}
review = study.finalize(conclusion="b wins")     # -> review.json
print(cmp["best"], "->", review["study_id"])
```

Per-experiment `report.json` (flat):

```json
{
  "id": "exp-...", "name": "b",
  "spec": {"name": "b", "train": {"mode": "stub", "learning_rate": 5e-05}, "eval": {"provider": "stub"}},
  "status": "completed",
  "steps": [{"name": "create", "...": "..."}, {"name": "train", "...": "..."}, {"name": "eval", "...": "..."}, {"name": "finalize", "...": "..."}],
  "dataset": {"path": ".../experiments/b/dataset", "format": null, "rows": 8},
  "model":   {"id": "model-...", "path": ".../experiments/b/model", "base_model": null, "finetuning_type": null, "mode": "stub"},
  "eval":    {"id": "eval-...", "score": 0.2, "metrics": {}, "path": ".../experiments/b/eval"}
}
```

Study-level `review.json`:

```json
{
  "study_id": "lr-smoke", "goal": "compare two stub configs", "n_experiments": 2,
  "varied_keys": ["learning_rate"],
  "experiments": [{"name": "a", "score": 0.1, "learning_rate": 0.0001}, {"name": "b", "score": 0.2, "learning_rate": 5e-05}],
  "ranking": [{"name": "b", "score": 0.2}, {"name": "a", "score": 0.1}],
  "best": {"name": "b", "score": 0.2},
  "conclusion": "b wins", "status": "completed"
}
```

## Running experiments in parallel (your job, not DataOS's)

The study is just a directory, and `study.compare()` reads each experiment's
`report.json` off disk — so the API is identical whether you run experiments in
one `for` loop or fan them out. To parallelize, launch each experiment in its own
process pinned to a GPU, e.g. `CUDA_VISIBLE_DEVICES=k python run_one.py --study
lr-smoke --spec b`, where each worker does `dataos.study("lr-smoke")` (reopens) →
`study.experiment(spec)` → train/eval/finalize. When all workers finish, call
`study.compare()` / `study.finalize()` from the main process. DataOS neither
schedules nor knows about the parallelism — assigning GPUs and deciding the
degree of concurrency is entirely yours.

## Boundaries

- **You are the runtime/planner. No embedded agent, no LLM key, no scheduler.**
  DataOS ships no planning agent, prompts, LLM client, or GPU scheduler — the
  whole pipeline is your Python calls. Reaching for an API key to run it means you
  are doing it wrong.
- **No one-shot `run()`; parallelism is yours.** Call the stages yourself; DataOS
  owns no orchestration loop and no resource scheduler.
- **Operators come from the external `dataos-operator` package only.** Import from
  `dataos_operator` or discover via `dataos.catalog()`. DataOS ships no operators.
- **Datasets are `dataos_operator.Dataset`, not a DataOS type.** `dataos.load`
  returns one; operators consume/return one; `exp.train` takes one.
- **Train/eval are open-config executors.** Pass any LLaMA-Factory / lm-eval knob
  through `config` (or `spec["train"]`/`spec["eval"]`) — no allowlist. DataOS only
  reserves the keys it must own (step 8) and validates the required ones.
- **Stub mode is the no-GPU contract.** `train`: `config=None`/`spec` absent **or**
  `{"mode": "stub"}` → stub. `eval`: stub must be **explicit** — pass
  `config={"provider": "stub", "provider_config": {"score": ...}}` (or put it in
  `spec["eval"]`); a bare `exp.eval()` with no config and no `spec["eval"]` raises.
  Stub needs only `datasets` + `dataos-operator`; real `llamafactory` /
  `lm_eval_native` need the `train` / `eval` extras and a GPU.
- **Name your experiments.** Each `spec["name"]` must be distinct within a plan —
  it is the directory slug and the key `compare()` joins on.
- **Verify the real files.** Read `experiments/<name>/report.json` and
  `review.json`; do not fabricate the summaries.
- **Keep training fields to known shapes.** Alpaca: `instruction`/`output`
  (+ optional `input`/`system`/`history`). ShareGPT: `conversations`
  (+ optional `system`/`tools`). Unexpected non-string values in required fields
  make the train stage raise.
