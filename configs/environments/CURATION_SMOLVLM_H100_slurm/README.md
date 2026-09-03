# Curation-Bench / SmolVLM H100 deployment

This deployment is intentionally separate from `PTB_H100_slurm`. It provisions the first
supported Curation-Bench target, `llava665k_smolvlm_8bench_10k_unlimited`, without changing any
PostTrainBench runtime or asset.

The deployment has five frozen pieces:

- DataOS control venv: `envs/dataos_cli/.venv` (only the DataOS lifecycle CLIs).
- Pinned Curation-Bench checkout and its three upstream venvs.
- Machine-local profile mapping the exact LLaVA-665K pool, SmolVLM base model and LMUData.
- A login-node ChatAnywhere judge; compute nodes only generate frozen predictions.
- `assets.yaml`, which pins public artifact revisions and all eight VLMEvalKit TSV checksums.

The pinned Curation-Bench snapshot omits `vlmeval/utils/result_transfer.py` even though its
`run.py` imports it unconditionally. `compat/result_transfer.py` is the exact file from the pinned
upstream VLMEvalKit revision declared in `assets.yaml`; `build.sh eval` installs it, and the
`assets.py verify` command checks its SHA-256. It does not run for this deployment's
`MMMU_DEV_VAL` task, but the module must exist for VLMEvalKit to start.

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

The official run requests two H100s for training and evaluation; the frozen trainer derives two
DDP processes from the Slurm allocation. CPU curation retains the cluster's mandatory minimum GPU.
Evaluation is split at the task contract: the Slurm job runs VLMEvalKit `--mode infer` with no
credentials, then
`dataos-eval collect` runs `--mode eval --reuse` on the login node with ChatAnywhere
`qwen3.5-plus`. The endpoint and key-file path are environment facts; model and concurrency are
frozen task facts. The credential-log compatibility patch is verified before use so the API key
cannot be written by VLMEvalKit's INFO logging.
