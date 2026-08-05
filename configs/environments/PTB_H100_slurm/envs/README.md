# PTB_H100_slurm uv environments

Five isolated uv-managed venvs. The three frozen per-stage baselines are deliberately separate
because their CUDA/torch/deps conflict (sft: py3.12 torch2.8/cu129 · rl: py3.11 torch2.6/cu124 ·
eval: py3.12 torch2.9/cu128); on top of those, two DataOS envs: `dataos_sft` (sft baseline + the
repo's own tools installed from the local checkout — env.yaml's `sft_env` points at it) and
`train-cli` (thin orchestrator: pydantic + dataos-train only). Each is a standard venv
(`.venv/bin/python`); uv is just the fast builder + version pinner.

Build on the exec host (uv installed at ~/.local/bin/uv):
```bash
cd configs/environments/PTB_H100_slurm/envs
./build.sh eval && ./build.sh sft && ./build.sh rl && ./build.sh dataos_sft && ./build.sh train-cli
```
`env.yaml.tools.<env>.path` points at each `.venv`. Everything is called by `{path}/bin/<tool>`
(no conda, no activate needed). Version pins match the previously-verified conda envs
(Llama-op020 / lm-eval / dataos-verl).

The venvs DataOS uses are built inside the deploy checkout (`configs/environments/PTB_H100_slurm/
envs/*/.venv`, git-ignored) — except the rl/eval baselines, which still live at the legacy
`/data/hcgu/users/tjh/dataos_runtime/...` location (absolute paths in env.yaml).

flash-attn: no prebuilt wheel exists for these torch combos and the system nvcc (13.2) mismatches
torch's CUDA — build.sh reuses the exact-matching flash_attn from the source conda env (same
py/torch/cuda/abi). Without it the env still runs (sdpa attention), just slower.

NOT here (in the three frozen baselines): `dataos_operator` / `dataos_train` — DataOS's private
tools live in the **derived** `dataos_sft` env instead (sft baseline + `tools/train` +
`tools/operator` installed from the local checkout). Their versions are pinned by the
`dataos_tools` declaration in `env.yaml` (a test keeps declaration == `tools/*/pyproject`
version); upgrade = bump the version, update the declaration, run `scripts/sync_envs.py`.
