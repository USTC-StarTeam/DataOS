# h100_slurm uv environments

Three isolated uv-managed venvs, one per stage — deliberately separate because their
CUDA/torch/deps conflict (sft: py3.12 torch2.8/cu129 · rl: py3.11 torch2.6/cu124 ·
eval: py3.12 torch2.9/cu128). Each is a standard venv (`.venv/bin/python`); uv is just the
fast builder + version pinner.

Build on the exec host (uv installed at ~/.local/bin/uv):
```bash
cd environments/h100_slurm/envs
./build.sh eval && ./build.sh sft && ./build.sh rl
```
`env.yaml.tools.<env>.path` points at each `.venv`. Everything is called by `{path}/bin/<tool>`
(no conda, no activate needed). Version pins match the previously-verified conda envs
(Llama-op020 / lm-eval / dataos-verl).

flash-attn: no prebuilt wheel exists for these torch combos and the system nvcc (13.2) mismatches
torch's CUDA — build.sh reuses the exact-matching flash_attn from the source conda env (same
py/torch/cuda/abi). Without it the env still runs (sdpa attention), just slower.

NOT here: `dataos_operator` — that's DataOS's private method library, brought at run time via
PYTHONPATH from deploy_root; these envs only carry general frameworks.
