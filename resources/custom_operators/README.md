# custom_operators — project-local DataOS operators

Operators here extend the `dataos-operator` catalog **without editing or reinstalling
the package**. They are discovered because `DATAOS_OPERATOR_PLUGINS` points at this
directory (set in the agent-host env and exported by the PLAYBOOK before running a
recipe on H100). Every top-level `.py` file is scanned for `DataOperator` subclasses,
which then appear in `dataos_operator.catalog()` / `usage()` and can be used by name in
recipes exactly like a built-in operator.

## Why this exists

When a sub-agent hits a task the 31 built-in operators can't do (e.g. "keep only rows
whose `output` is a valid function call"), it hand-writes the logic. If that logic is
general and worth reusing, it **sediments it here as a proper operator** — so the
main-agent can plan it by name next time and it is reused across runs. The toolbox gets
richer the more DataOS runs.

## How to add one

Read the framework's own guide for the exact contract:

```bash
python -c "import dataos_operator as o; print(o.guide())"   # "Writing your own operator" section
```

In short: subclass the family base (`SelectorOperator` / `MapperOperator` /
`EvaluatorOperator` / …) from `dataos_operator.core.contracts`, set a class attribute
`operator_name`, write a docstring (one-line summary + `Implementation:` + `Dependency:`),
and implement the family's one abstract method (`select` / `map_record` / …). Drop the
`.py` file here. Files starting with `_` are treated as helper modules (not scanned); a
name that collides with a built-in is ignored; an import-broken file is skipped.

## Provided

- **`function_call_filter.py`** — `function_call_filter` (selector): keep only rows whose
  `output` parses as a valid `[{"name","arguments"}]` function-call list. The operator BFCL
  needed — reuse it for tool-calling benches instead of re-writing a filter.
