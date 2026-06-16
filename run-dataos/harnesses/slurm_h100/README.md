# H100 Slurm remote-train harness (agent-version-ptb)

Ships **training** to an H100 Slurm cluster while data + eval stay on the local box. Used by
`dataos/train/slurm_h100.py` when a train spec carries `executor: "slurm_h100"`.

## Flow (runs inside `run_train` on the local box)
```
export dataset -> train.jsonl  →  rsync (train_sft.py + train.jsonl + run_train.sbatch) to H100
→  sbatch (gpu partition) + poll sacct  →  rsync LoRA adapter back  →  return finetuning_type=lora Model
```
The returned Model has the adapter in its dir + `meta.base_model` (the logical HF id), so the
downstream local eval treats it like any locally-trained LoRA.

## Files
- `train_sft.py` — self-contained PEFT LoRA SFT (alpaca jsonl / messages); writes the adapter.
- `run_train.sbatch.tmpl` — sbatch template; `{placeholders}` filled by the backend via literal
  `.replace` (the template also contains shell `{ ...; }` groups, so do NOT `str.format` it).

## One-time H100 setup (as the run user, e.g. tangjiahui)
- Dedicated env at `/data/tangjiahui/dataos_train/.venv` (torch cu124 + peft + transformers +
  datasets + accelerate), built on the login node (has internet); gpu nodes reuse it offline.
- Base models from `/data/share_weight/<model>` (e.g. Llama-3.2-3B-Instruct), or a path under
  the run user's `hf_cache`.
- Local box needs a working `ssh h100` (dedicated key authorized in H100 `~/.ssh/authorized_keys`).

## Config knobs (train spec)
`executor: "slurm_h100"`, `model_name_or_path` (H100-local base path), `base_model_id` (HF id
for local eval), hyperparams (`num_train_epochs`, `per_device_train_batch_size`,
`gradient_accumulation_steps`, `learning_rate`, `lora_rank`, `lora_alpha`, `cutoff_len`,
`max_samples`), and an optional `slurm` block (`host`, `remote_root`, `venv`, `hf_home`,
`partition`, `qos`, `time`, `poll_interval_s`, `timeout_s`). Defaults target `partition=gpu`,
`qos=gpu_normal`; use `partition=debug`/`qos=debug_normal` for quick smokes.

## Verified
2026-06-16: DataOS on dazu5090 (RTX 5090) → data local → train submitted to H100 (job 5991,
debug, Llama-3.2-3B LoRA) → adapter pulled back → Model(finetuning_type=lora). End-to-end green.
