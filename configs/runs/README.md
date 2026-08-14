# Run configs

One JSON per run. Copy `_template.json`, fill it, launch it. Everything not in this file resolves
through `environment` → `configs/environments/<env>/env.yaml` and `task` →
`configs/tasks/<task>/`.

## The rule this directory exists to keep

**`env.yaml` describes the machine. A run config chooses.** A value that could differ between two
runs on the same cluster belongs here, never there — the alternative is what happened to `qos`,
which sat in `env.yaml` with no run-level override, so switching QOS meant maintaining a second
environment file identical in 79 of 81 lines. Constraints are not here either: they are prompt
text (`src/dataos/prompt/*/context/cluster_discipline.yaml`), stated once and golden-tested,
derived from the values below.

## `exec.resource` — the four choices

| field | required | what it means |
|---|---|---|
| `gpus` | yes | cards per job. Also what the eval job asks for — they travel together now. |
| `cpus` | yes | CPUs per task. The partition refuses more than `cpus_per_gpu` × `gpus`. |
| `qos` | **yes** | which QOS to buy with. Must be one of `env.slurm.qos_available`; a typo is rejected at load rather than PENDing forever with no error. `gpu_normal` is the guarded one — choosing it makes the GPU-Guard prescription appear in the experimenter's prompt automatically. |
| `node` | no | pin to one node. `null` lets Slurm pick from `env.slurm.nodes`. Honoured by training, data and eval alike; before 2026-08-14 eval ignored it and 68 of 94 eval jobs of a gpu02-pinned run landed elsewhere. |

## `budget.eval_samples`

`-1` full benchmark · `-2` AUTO (the loop sizes each node's eval) · a positive integer to cap.
A subset score is not comparable with a full one, so whatever you choose here shows up next to
every score as its caliber.

## Naming

`workspace` is the run's identity everywhere — the workspace dir, the run root on the exec host,
the archive. Use `workspace/ptb-<bench>-<batch>` and keep `monitor_port` unique across runs you
intend to run at the same time.

Pre-2026-08-14 configs are in `_archive-20260814/`. They still load; none of them is a template.
