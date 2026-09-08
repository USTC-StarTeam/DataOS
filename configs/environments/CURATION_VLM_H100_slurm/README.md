# Curation-Bench Qwen2.5-VL-3B / LLaVA-1.5-7B

Independent task/model profile for the two official LLaVA-665K 10K targets.
The pinned harness, DataOS CLI venv, dataset and evaluation data are shared read-only with
the SmolVLM deployment; no environment rebuild is needed. Each run owns its own suite,
curation worktrees and checkpoints. PTB and the existing SmolVLM profile are unchanged.

`assets.yaml` pins the public Qwen2.5-VL-3B-Instruct and llava-1.5-7b-hf snapshots.
Do not substitute a trained baseline checkpoint or the separately reconstructed LLaVA init.
Machine-local `env.yaml` and `profile.yaml` stay untracked. Install the latter as the provider's
`profiles/dataos-h100-vlm.yaml`. The profile maps both model keys to the dedicated asset root.

Verify each task on H100 before launching (substitute its task ID and absolute profile path):

```bash
configs/environments/CURATION_SMOLVLM_H100_slurm/envs/dataos_cli/.venv/bin/python \
  configs/environments/CURATION_SMOLVLM_H100_slurm/assets.py verify \
  --manifest configs/environments/CURATION_VLM_H100_slurm/assets.yaml \
  --harness-root /data2/hcgu/dataos_curation_bench \
  --profile /data2/hcgu/dataos_curation_bench/profiles/dataos-h100-vlm.yaml \
  --task llava665k_qwen_8bench_10k_unlimited
```

Run configs `curation-qwen25vl3b-h100-max.json` and `curation-llava7b-h100-max.json` request
11 official iterations, two H100 GPUs on gpu01/gpu_priv, and Codex gpt-5.6-sol at max effort.
Training remains owned by the official harness. Evaluation resolves the model adapter from
the active official task's model key through the pinned harness registry.

## Qwen mixed-modality compatibility

The upstream Qwen trainer at the pinned revision sends null images to qwen-vl-utils. Our fixed
Arrow pool contains text-only rows, including in the unmodified random baseline. The maintained
patch normalizes None, an empty list and null list entries to no image content. It does not drop
rows, add dummy images, change text/labels, or modify learning rate, epochs or batch/accumulation.
DDP must detect unused vision parameters on text-only microbatches; non-reentrant gradient
checkpointing keeps that distributed path supported. These two runtime flags are recorded as
compatibility changes, not claimed to be byte-identical upstream training.

Apply only to the exact pinned trainer; the script preserves an original-file backup and refuses
unknown revisions or unrelated edits:

```bash
python configs/environments/CURATION_VLM_H100_slurm/compat/patch_qwen_text_only.py \
  --project /data2/hcgu/dataos_curation_bench/vendor/curation-train
```

`assets.yaml` records both hashes; asset verification requires the patched Qwen hash. SmolVLM,
LLaVA, the data pool, official task YAMLs and frozen scoring definitions are not changed.
