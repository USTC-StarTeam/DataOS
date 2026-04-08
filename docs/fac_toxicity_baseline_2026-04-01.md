# FAC Toxicity Baseline

Date: 2026-04-01

## Setup

- Runner: `dataos/eval` native provider with `transformers` adapter
- Task: `toxicity_detection`
- Dataset: ToxicChat `0124` test split (`5083` examples, `362` toxic, `7.12%` positive rate)
- Generation config: `batch_size=16`, `max_new_tokens=4`, `do_sample=False`
- Remote code root: `/home/hcgu/dataos-gsm8k-remote-test`
- Remote dataset root used by DataOS eval: `/home/hcgu/data/fac_synthesis_datasets/hf_datasets/lmsys__toxic-chat__0124`

The raw ToxicChat CSV snapshot from the FAC prep thread was first converted to a Hugging Face
`DatasetDict` saved with `save_to_disk`, so the run stayed inside the repository's fixed
`DatasetArtifact` contract.

## Results

| Model | F1 | Precision | Recall | Accuracy | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mistralai/Mistral-7B-Instruct-v0.3` | `0.6716` | `0.7229` | `0.6271` | `0.9563` | `0` |
| `Qwen/Qwen2-7B-Instruct` | `0.6627` | `0.5860` | `0.7624` | `0.9447` | `1` |
| `LLM-Research/Meta-Llama-3.1-8B-Instruct` | `0.5700` | `0.6944` | `0.4834` | `0.9481` | `7` |
| OpenAI moderation reference (`any score >= 0.02`) | `0.6141` | `0.5476` | `0.6989` | `0.9374` | `0` |

Ranking by F1:

1. `mistralai/Mistral-7B-Instruct-v0.3`
2. `Qwen/Qwen2-7B-Instruct`
3. `LLM-Research/Meta-Llama-3.1-8B-Instruct`

## Notes

- Mistral and Qwen both beat the dataset's bundled OpenAI moderation reference on F1.
- Qwen favored recall more strongly than Mistral, with more false positives.
- Llama 3.1 underperformed the OpenAI moderation reference on recall and had a few instruction-following failures.
- The invalid outputs were formatting issues, not parser bugs. Examples included responses like
  `I'm sorry,` or partial structured outputs instead of a strict `0` / `1` label.

## Remote Artifacts

- Qwen eval artifact:
  `/home/hcgu/dataos-gsm8k-remote-test/fac-toxicity-baseline-runs/qwen2-7b-instruct/sandbox/artifacts/evals/eval-20260401133605494450-ca32157f`
- Mistral eval artifact:
  `/home/hcgu/dataos-gsm8k-remote-test/fac-toxicity-baseline-runs/mistral-7b-instruct-v0-3/sandbox/artifacts/evals/eval-20260401133545893402-dc7c35f7`
- Llama eval artifact:
  `/home/hcgu/dataos-gsm8k-remote-test/fac-toxicity-baseline-runs/llama-3-1-8b-instruct/sandbox/artifacts/evals/eval-20260401133536815452-56b8c876`

Matching local summary dump:

- `docs/fac_toxicity_baseline_2026-04-01_summary.json`
