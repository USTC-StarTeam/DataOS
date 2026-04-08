# FAC Synthesis Thread Summary

Date: 2026-04-01

## What was done

This thread focused on `Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs` and preparing the main assets needed to reproduce or extend the paper's baselines.

Completed:

- Read the paper and extracted the main downstream tasks:
  - toxicity detection
  - reward modeling
  - behavior steering
  - instruction following
- Matched the core datasets mentioned in the paper and downloaded the Hugging Face-hosted ones to the remote 5090 machine.
- Matched the backbone models used in the paper and downloaded them to the remote 5090 machine.

## Remote machine

- SSH alias: `bdaa_5090-1`
- Remote root used in this thread: `/home/hcgu/data`

## Datasets downloaded

Remote dataset root:

- `/home/hcgu/data/fac_synthesis_datasets`

Saved manifest:

- `/home/hcgu/data/fac_synthesis_datasets/manifest.json`

Downloaded dataset directories:

- `/home/hcgu/data/fac_synthesis_datasets/hub_datasets/trl-lib__hh-rlhf-helpful-base`
- `/home/hcgu/data/fac_synthesis_datasets/hub_datasets/Anthropic__hh-rlhf`
- `/home/hcgu/data/fac_synthesis_datasets/hub_datasets/nvidia__HelpSteer2`
- `/home/hcgu/data/fac_synthesis_datasets/hub_datasets/lmsys__toxic-chat`
- `/home/hcgu/data/fac_synthesis_datasets/hub_datasets/allenai__reward-bench`
- `/home/hcgu/data/fac_synthesis_datasets/hub_datasets/Anthropic__model-written-evals`
- `/home/hcgu/data/fac_synthesis_datasets/hub_datasets/HuggingFaceH4__ultrachat_200k`

Approximate sizes:

- `trl-lib__hh-rlhf-helpful-base`: `25.8 MB`
- `Anthropic__hh-rlhf`: `90.4 MB`
- `nvidia__HelpSteer2`: `38.8 MB`
- `lmsys__toxic-chat`: `58.1 MB`
- `allenai__reward-bench`: `14.8 MB`
- `Anthropic__model-written-evals`: `71.3 MB`
- `HuggingFaceH4__ultrachat_200k`: `1548.8 MB`

Download source notes:

- These were downloaded with `HF_ENDPOINT=https://hf-mirror.com`.
- The behavior steering CAA dataset itself was not downloaded in this thread.
- The extra Figure 1 comparison instruction datasets were also not downloaded in this thread.

## Models downloaded

Remote model root:

- `/home/hcgu/data/fac_synthesis_models/models`

Saved manifests:

- `/home/hcgu/data/fac_synthesis_models/manifest.json`
- `/home/hcgu/data/fac_synthesis_models/remaining_manifest.json`

Downloaded model directories:

- `/home/hcgu/data/fac_synthesis_models/models/LLM-Research__Meta-Llama-3.1-8B-Instruct`
- `/home/hcgu/data/fac_synthesis_models/models/LLM-Research__Meta-Llama-3-8B`
- `/home/hcgu/data/fac_synthesis_models/models/mistralai__Mistral-7B-Instruct-v0.3`
- `/home/hcgu/data/fac_synthesis_models/models/Qwen__Qwen2-7B-Instruct`

Approximate sizes:

- `LLM-Research__Meta-Llama-3.1-8B-Instruct`: `29.93 GB`
- `LLM-Research__Meta-Llama-3-8B`: `14.97 GB`
- `mistralai__Mistral-7B-Instruct-v0.3`: `27.00 GB`
- `Qwen__Qwen2-7B-Instruct`: `14.20 GB`

Source notes:

- `Meta-Llama-3.1-8B-Instruct` was downloaded from ModelScope.
- `Meta-Llama-3-8B` was downloaded from ModelScope.
- `Mistral-7B-Instruct-v0.3` and `Qwen2-7B-Instruct` were downloaded through `hf-mirror`.

Important note about `Meta-Llama-3.1-8B-Instruct`:

- The first ModelScope run downloaded both the needed `safetensors` files and an extra `original/consolidated.00.pth`.
- That means this directory contains redundant weight material and is larger than necessary.
- If disk space matters, inspect:
  - `/home/hcgu/data/fac_synthesis_models/models/LLM-Research__Meta-Llama-3.1-8B-Instruct/original/consolidated.00.pth`
- Delete it only if the downstream code does not rely on the `original/` files.

## Paper mapping used in this thread

Main task backbones from the paper:

- `Llama-3.1-8B-Instruct`
- `Mistral-7B-Instruct`
- `Qwen2-7B-Instruct`

Additional model downloaded because the official instruction-following script uses it:

- `Meta-Llama-3-8B`

Version note:

- The paper text names `Mistral-7B-Instruct` without an explicit version.
- This thread mapped it to the official `mistralai/Mistral-7B-Instruct-v0.3`.

## Suggested next steps in a new window

- Verify no more downloads are running:
  - `ssh bdaa_5090-1 "pgrep -af download_remaining_models.py || true"`
- Check remote disk usage:
  - `ssh bdaa_5090-1 "df -h /home/hcgu"`
- Decide whether to remove the redundant `Meta-Llama-3.1-8B-Instruct/original/consolidated.00.pth`.
- Download the missing behavior steering CAA data if the new baseline run needs that task.
- Download Figure 1 comparison datasets if the new baseline run needs the instruction-following efficiency comparison rather than only the paper's main four tasks.

## Sanity-check commands

List datasets:

```bash
ssh bdaa_5090-1 "ls -1 /home/hcgu/data/fac_synthesis_datasets/hub_datasets"
```

List models:

```bash
ssh bdaa_5090-1 "ls -1 /home/hcgu/data/fac_synthesis_models/models"
```

Check model sizes:

```bash
ssh bdaa_5090-1 "python3 - <<'PY'
from pathlib import Path
root = Path('/home/hcgu/data/fac_synthesis_models/models')
for p in sorted(root.iterdir()):
    total = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    print(f'{p.name}\\t{total/1024/1024/1024:.2f} GB')
PY"
```
