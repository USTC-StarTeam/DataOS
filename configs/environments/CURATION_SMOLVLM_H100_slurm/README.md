# Curation-Bench / SmolVLM H100 deployment

This deployment is intentionally separate from `PTB_H100_slurm`. It provisions the first
supported Curation-Bench target, `llava665k_smolvlm_8bench_10k_unlimited`, without changing any
PostTrainBench runtime or asset.

The deployment has four frozen pieces:

- DataOS control venv: `envs/dataos_cli/.venv` (only the DataOS lifecycle CLIs).
- Pinned Curation-Bench checkout and its three upstream venvs.
- Machine-local profile mapping the exact LLaVA-665K pool, SmolVLM base model and LMUData.
- `assets.yaml`, which pins public artifact revisions and all eight VLMEvalKit TSV checksums.

The real `env.yaml` and Curation profile contain machine coordinates and remain untracked. On the
H100 deployment they are:

```text
DataOS env       configs/environments/CURATION_SMOLVLM_H100_slurm/env.yaml
harness          /data2/hcgu/dataos_curation_bench
profile          /data2/hcgu/dataos_curation_bench/profiles/dataos-h100-smolvlm.yaml
asset root       /data2/hcgu/dataos_curation_assets
```

Build the software environments from the pinned checkouts:

```bash
cd configs/environments/CURATION_SMOLVLM_H100_slurm/envs
CURATION_ROOT=/data2/hcgu/dataos_curation_bench \
FLASH_ATTN_DONOR=/data/hcgu/users/tjh/dataos_runtime/environments/PTB_H100_slurm/envs/rl/.venv \
  ./build.sh all
```

The donor is safe because both environments use Python 3.11, torch 2.6.0+cu124 and
flash-attn 2.7.4.post1. The build verifies those facts before copying the binary extension.

Download the pinned LLaVA Arrow snapshot and SmolVLM model on a networked host, then copy their
directories to the paths in the profile. The H100 login node cannot resolve Hugging Face:

```bash
hf download Ethlake/llava-665k --repo-type dataset \
  --revision 235a8adf266bb6dc02a099dc0221d28dec058f54 --local-dir <llava665k-dir>
hf download HuggingFaceTB/SmolVLM-Base \
  --revision a576ab7b4354916a0cb5b58a3bd973a9a526ee7c --local-dir <smolvlm-dir>
```

LMUData is fetched on the login node and checked against the exact hashes in the pinned
VLMEvalKit checkout:

```bash
envs/dataos_cli/.venv/bin/python assets.py prefetch-eval \
  --asset-root /data2/hcgu/dataos_curation_assets
```

Before starting a DataOS run, validate the selected official task and every transitive asset:

```bash
envs/dataos_cli/.venv/bin/python assets.py verify \
  --harness-root /data2/hcgu/dataos_curation_bench \
  --profile /data2/hcgu/dataos_curation_bench/profiles/dataos-h100-smolvlm.yaml \
  --task llava665k_smolvlm_8bench_10k_unlimited
```

This environment does not claim VisionFlan, LLaVA-7B or Qwen task support. Add a separate pinned
deployment after those pools/models are provisioned; do not point their task IDs at placeholder
paths in this profile.
