# PostTrainBench Task Assets

This directory contains DataOS-maintained lm-evaluation-harness task assets for
PostTrainBench protocol reproduction.

The current scope is intentionally narrow: only GSM8K is included. The goal is
to make the first protocol auditable before adding more PostTrainBench tasks.

## Current Protocol

`posttrainbench_gsm8k.yaml` defines the GSM8K task used by the verification
runner.

- Dataset: `openai/gsm8k`
- Dataset name: `main`
- Split: `test`
- Output type: `generate_until`
- Few-shot examples: 10 examples from the train split, shuffled with seed 42
  and inserted through the runner as `system_instruction`
- lm-eval `num_fewshot`: `0`, because the 10 examples are injected explicitly
  to match the PostTrainBench chat protocol
- Metric: `accuracy`
- Metric implementation: `utils.process_results`
- Answer rule: numeric end-match, compatible with PostTrainBench's Inspect AI
  numeric match behavior
- Verification runner max generation length: `max_gen_toks=4000`

The runner selects model-family chat templates from `templates/`:

- `qwen3.jinja`
- `smollm.jinja`
- `gemma3.jinja`

Changing any template, prompt rule, extraction rule, seed, stop sequence, or
generation setting changes the protocol and must be treated as a protocol
version update.

## Maintenance Rules

1. Keep benchmark-specific logic in this directory or in a dedicated runner.
   The generic `lm_eval_native` backend should remain benchmark-agnostic.
2. Add one PostTrainBench benchmark at a time.
3. For each new benchmark, include the task YAML, helper functions, template
   requirements, and a short validation note.
4. Record reference model results before treating a protocol as aligned.
5. Preserve `protocol.json` artifacts from validation runs outside the package
   when large sample outputs are involved. Do not commit large run artifacts or
   model outputs to the repository.

## Known Limitations

The current GSM8K package is a working protocol bridge, not complete
PostTrainBench coverage. It has been validated strongly enough to show that the
DataOS native lm-eval execution path is viable, but SmolLM3 and Gemma3 still
show small remaining gaps against the current reference numbers. Future work
should compare sample-level outputs and confirm generation configuration,
template details, and checkpoint revisions.

