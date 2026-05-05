# Stable Operator Library V1

This document describes the first hand-written stable operator batch under `dataos/operators/`.
It is the operator-engineer-facing companion to `docs/dataos_overview.md`.

## Design Position

These operators are intentionally **not** a direct copy of DataJuicer or DataFlow.
They are DataOS-native in three ways:

1. They are **artifact-first**: every operator consumes `DatasetArtifact` and emits a fresh `DatasetArtifact` with lineage.
2. They are **split-aware**: `DatasetDict` is the default mental model, not an afterthought.
3. They are **research-loop-oriented**: metadata, logs, and deterministic behavior matter as much as the transform itself.

The goal is not to ship the biggest operator zoo. The goal is to ship a compact, auditable, reproducible operator set that works well inside DataOS' daemon-style research loop.

## Families

### 1. Repair / Governance

| Operator | Purpose | DataOS-specific trait |
|---|---|---|
| `schema_guard` | Normalize heterogeneous SFT rows into canonical `instruction/input/response` fields. | Treats schema repair as a first-class stable operator instead of hidden pre-processing glue. |
| `pii_redact` | Redact obvious emails / phones / URLs from text columns. | Preserves dataset shape and logs redaction counts into artifact metadata for later audit. |

### 2. Cheap Filters

| Operator | Purpose | DataOS-specific trait |
|---|---|---|
| `length_filter` | Drop rows outside a simple length band. | Supports total-length and per-column thresholds while staying HF-dataset-native. |
| `lang_filter` | Keep only rows whose dominant script matches allowed labels such as `en`, `zh`, or `mixed`. | Uses a lightweight split-aware script heuristic instead of depending on an external service. |
| `quality_filter` | Remove weak instruction-response pairs via heuristic scoring or optional LLM judging. | Keeps an explicit `quality_judgments.jsonl` audit trail beside the artifact. |

### 3. Annotation / Balancing / Ordering

| Operator | Purpose | DataOS-specific trait |
|---|---|---|
| `category_tag` | Assign coarse task-surface labels such as `translation`, `math`, `code`, `writing`. | Designed to create control signals for later balancing and analysis operators. |
| `category_balance` | Rebalance a split by category using deterministic downsample / upsample strategies. | Interleaves categories after balancing so the resulting split is not blocky. |
| `curriculum_sort` | Order rows from easy-to-hard or hard-to-easy using a simple surface complexity proxy. | Emits an explicit curriculum score column that downstream analysis can inspect. |

### 4. Data Construction / Mixing

| Operator | Purpose | DataOS-specific trait |
|---|---|---|
| `instruction_synthesis_template` | Create deterministic instruction rewrites while preserving the gold response. | Adds provenance columns such as `dataos_synth_template` and `dataos_synth_source_index`. |
| `mixture` | Build a training mixture from multiple upstream dataset artifacts. | Operates on artifact ids directly and writes source provenance into every mixed row. |

## Operator Summary

### `schema_guard`

- Normalizes common aliases like `prompt -> instruction` and `output -> response`.
- Optional strict mode can fail fast instead of silently repairing.
- Best used near the front of a pipeline when data sources are heterogeneous.

### `pii_redact`

- Supports email, phone, and URL redaction.
- Can either mask PII in-place or drop rows containing PII.
- Useful before synthesis, export, or external judging.

### `length_filter`

- Supports `words` or `chars` units.
- Can enforce total-length bounds and per-column bounds.
- Useful for trimming pathological long-tail data before training or judging.

### `lang_filter`

- Supports `allowed_languages=[...]`, e.g. `['en']`, `['zh']`, `['mixed']`.
- Optional annotation mode can attach the detected label to a column.
- Intended as a cheap routing filter, not a benchmark-grade language detector.

### `quality_filter`

- `mode='heuristic'`: fully local, deterministic quality scoring.
- `mode='llm_judge'`: calls an OpenAI-compatible `/chat/completions` endpoint with JSON output.
- In LLM mode, use standard arguments such as `base_url`, `api_key`, and `model`.

### `category_tag`

- Uses a simple task-surface preset instead of heavyweight semantic classification.
- Creates a low-cost structural signal that other operators can consume.
- Especially useful before `category_balance`.

### `category_balance`

- Supports `downsample_to_min`, `cap`, and `upsample_to_max`.
- Balances one split at a time, typically `train`.
- Keeps other splits untouched.

### `curriculum_sort`

- Computes a simple surface complexity proxy from prompt/response length, formatting, and code markers.
- Designed for deterministic curriculum experiments rather than perfect difficulty estimation.
- Writes the resulting score into a configurable column.

### `mixture`

- Mixes multiple parent artifacts by artifact id.
- Adds `dataos_source_artifact` provenance to every mixed row.
- Keeps the operator small and deterministic instead of turning it into a generic recipe language.

### `instruction_synthesis_template`

- Generates small, deterministic instruction variants from a seed training split.
- The first version deliberately avoids hidden LLM costs.
- Intended as a cheap baseline against which future `self_instruct_lite` or candidate operators can be compared.

## Why this is different from DataJuicer / DataFlow

### Compared with DataJuicer

- DataJuicer emphasizes a broad, reusable operator zoo across many data-engineering scenarios.
- DataOS stable operators are narrower and more opinionated: they optimize for SFT research loops, artifact lineage, and split-aware experimentation.
- DataOS operators record compact but explicit run metadata because the downstream consumer is not only the transformed data, but also the analyzer / critic / report loop.

### Compared with DataFlow

- DataFlow-style recipes often treat data steps as workflow nodes in a broader pipeline system.
- DataOS stable operators are deliberately smaller-grain and plug directly into the `run_task()` operator-plan stage.
- The key abstraction in DataOS is the typed artifact chain, not a generic DAG DSL.

## Suggested Pipeline Motifs

### Minimal cleanup motif

1. `schema_guard`
2. `pii_redact`
3. `lang_filter`
4. `quality_filter`

### Category-aware motif

1. `schema_guard`
2. `category_tag`
3. `category_balance`
4. `curriculum_sort`

### Cheap augmentation motif

1. `schema_guard`
2. `quality_filter`
3. `instruction_synthesis_template`
4. `mixture`

## Notes for future work

- A future `self_instruct_lite` can be added on top of the same artifact contract, but should stay explicit about model cost and provenance.
- A future `quality_filter` judge pool can batch OpenAI-compatible requests more aggressively once the research daemon starts managing token backpressure.
- Candidate operators should remain separate from this stable library until validated and promoted.
