# DataOS Operator — Usage Guide

This guide is shipped inside the installed package. Call
`dataos_operator.usage()` to obtain it together with the auto-generated,
per-operator reference (which is introspected from the operator classes and
therefore always matches the installed version).

## Data model

Every operator consumes and produces a `Dataset` (an in-memory list of records;
Arrow/Hugging Face is an on-demand export format only). A record is a plain dict.

```python
from dataos_operator import Dataset

ds = Dataset.from_records([{"messages": [...], "metadata": {...}}, ...])
ds = Dataset.from_local("/path/to/data.jsonl")        # local jsonl
```

Dotted field paths (e.g. `metadata.domain`) are used throughout for keys.

## How operators are used

Every operator is one of two kinds, tagged `offline` or `online` in the
per-operator reference.

### Offline (static) — `op.run(dataset) -> dataset`

Pure, composable data transforms that run *before* training. This is every
mapper / pure-rule selector / mixer, plus the offline path of the policy
operators (which rank by a precomputed score key).

```python
from dataos_operator.operators.selector.length_filter import LengthFilter

ds = LengthFilter(text_key="text", min_length=3, max_length=2048).run(ds)
```

The data-centric idiom: an **evaluator** writes a score, then a
**score/ranking selector** keeps the best rows.

### Online (policy) — driven inside LLaMA-Factory training

The vendored LLaMA-Factory fork drives the runtime for you. Three steps:

1. Export the dataset to ShareGPT layout (writes `dataos_id` / `dataos_domain`):

   ```python
   from dataos_operator.adapters import export_sharegpt_dataset
   export_sharegpt_dataset(ds, bundle_dir, dataset_name="my_run", split_name="train")
   ```

2. Write a `dataos_runtime_config.json`:

   ```json
   {
     "mode": "online",
     "update_interval": 2,
     "train_dataset_paths": [".../train.jsonl"],
     "eval_dataset_paths":  [".../eval.jsonl"],
     "operators": [
       {"operator": "loss_selector", "kwargs": {"sample_size": 3000, "focus": "high"}}
     ],
     "backend": {"max_length": 2048, "max_new_tokens": 16}
   }
   ```

3. Train with the DataOS switch on:

   ```python
   run_exp({..., "dataos_enable": True,
            "dataos_config_path": ".../dataos_runtime_config.json",
            "dataos_update_interval": 2})
   ```

The backend reuses the training model itself to compute per-sample losses, so
no extra model is loaded. Operators are referenced by their **registry name**
(the `operator` field) — the same name shown in the per-operator reference.

> Under the hood the LLaMA-Factory controller drives the `PolicyRuntime` API
> (`prepare → after_forward → on_update_boundary → finalize`). You only need it
> directly to wire online operators into a *non*-LLaMA-Factory trainer.

### Extra inputs for some online selectors

- `less_selector`, `nice_selector` require `eval_dataset_paths` in the config.
- `near_selector`, `tsds_selector` require `query_dataset_paths`.

## Cost note

Online operators add a small per-step tax (per-sample loss bookkeeping; the
gradient-based selectors add periodic candidate-pool passes). They aim to win on
*data/steps-to-target*, not on raw per-step throughput. They are not currently
distributed-aware — drive them on a single data-parallel rank.

## Writing your own operator (local plugin)

When the built-in catalog lacks the operator you need — e.g. a "keep only rows
whose `output` is a valid function call" filter — **write it as a local operator
and drop it in a plugin directory**. No repo edit, no reinstall: point the env
var `DATAOS_OPERATOR_PLUGINS` at one or more directories (`os.pathsep`-separated).
Every top-level `.py` file there is scanned for `DataOperator` subclasses, which
then show up in `catalog()` / `usage()` and can be used by name exactly like a
built-in operator (in both the offline `.run()` path and online recipes).

### The recipe

1. **Subclass the family base** for your operator kind and implement its single
   abstract method:

   | kind | base class | implement | granularity |
   |---|---|---|---|
   | mapper | `MapperOperator` | `map_record(record) -> record` | per record |
   | evaluator | `EvaluatorOperator` | `evaluate_record(record) -> record` (writes a score field) | per record |
   | refiner | `RefinerOperator` | `refine_record(record) -> record` | per record |
   | reweighter | `ReweighterOperator` | `reweight_record(record) -> record` | per record |
   | selector | `SelectorOperator` | `select(datasets) -> dataset` | over dataset(s) |
   | mixer | `MixerOperator` | `mix(datasets) -> dataset` | over dataset(s) |
   | generator | `GeneratorOperator` | `generate(datasets) -> dataset` | over dataset(s) |
   | clusterer | `ClustererOperator` | `cluster(datasets) -> dataset` | over dataset(s) |

   (import them from `dataos_operator.core.contracts`.)

2. Set an **`operator_name`** class attribute — this is the registry name used in
   recipes. The **class docstring** becomes the catalog description; write it like
   the built-ins: one-line summary + `Implementation:` + `Dependency:` lines.
3. Take config via `**kwargs` in `__init__` and call `super().__init__(**kwargs)`;
   read record fields with `self.get_field(record, key, default)` /
   `self.require_field(record, key)` (dotted paths like `metadata.domain` work).

### Example — a function-call validity filter (the operator BFCL needed)

```python
# save as  <plugin_dir>/function_call_filter.py
import json
from collections.abc import Sequence

from dataos_operator.core.contracts import Dataset, SelectorOperator


class FunctionCallFilter(SelectorOperator):
    """Keep only rows whose ``output`` parses as a valid function-call list.

    Implementation: ``json.loads(output)`` must be a list of ``{"name","arguments"}``
    dicts; everything else (natural-language answers, "no access" negatives) is
    dropped. Use it before training on a tool-calling benchmark whose eval parses
    exact function-call syntax.
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "function_call_filter"

    def __init__(self, *, output_key: str = "output", **kwargs):
        super().__init__(output_key=output_key, **kwargs)
        self.output_key = output_key

    def select(self, data: Sequence[Dataset]) -> Dataset:
        rows = [record for dataset in data for record in dataset]
        return [r for r in rows if self._is_valid_call(self.get_field(r, self.output_key, ""))]

    @staticmethod
    def _is_valid_call(output: object) -> bool:
        if not isinstance(output, str):
            return False
        try:
            calls = json.loads(output)
        except (ValueError, TypeError):
            return False
        return isinstance(calls, list) and all(
            isinstance(c, dict) and "name" in c and "arguments" in c for c in calls
        )
```

### Use it

```bash
export DATAOS_OPERATOR_PLUGINS=/path/to/plugin_dir
```

```python
import dataos_operator
print(dataos_operator.catalog())        # function_call_filter is now listed

from function_call_filter import FunctionCallFilter
clean = FunctionCallFilter(output_key="output").run(ds)   # chains like any operator
```

Rules of thumb: files whose name starts with `_` are treated as helper modules
(not scanned for operators); a plugin whose `operator_name` collides with a
built-in is ignored; a plugin that fails to import is skipped without breaking the
rest of the catalog.
