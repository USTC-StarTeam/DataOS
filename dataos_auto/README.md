# dataos_auto — autonomous SFT runtime (Agent SDK)

The full-auto execution mode of DataOS, reimplemented as **owned Python code** driving
the deterministic `dataos` library (v0.2.5), with an LLM only at two judgment points
(via the [dataos-agent](https://github.com/USTC-StarTeam/DataOS-Agent) Claude SDK backend).

This replaces the skill's "main agent + nohup" autonomous mode. The loop, fan-out,
budget, stop-decision and resume are code now (runtime enforcement) — fixing the skill's
four pain points: orchestrator early-exit, fragile cold-resume, prose-only budget, and
manual network retry.

## Shape

```
run_auto(cfg):
  project.set_goal(criterion, wall_clock_budget)  # persisted for resume/provenance
  proj.load()                                     # rehydrate prior studies
  tree.load()                                     # best-over-project search state
  freeze_eval()                                  # fixed PostTrainBench protocol
  for each round until wall-clock is spent:
    tree.select_slots()                           # 2 exploit + 1 explore
    planner.next_plan()                           # LLM judgment point #1
    study.set_plan()
    subagent_exec.fan_out()                       # sequential exp sub-agents
      exp sub-agent: build data -> Slurm train -> frozen eval -> report.json
      missing report: resume the same Claude Code session
    reviewer.conclude()                           # LLM judgment point #2
    study.review()
  final_model := best scored node over the tree
```

## Stop policy

For the PTB branch, the hard stop condition is wall-clock budget:

```text
Budget(max_wall_clock_s=...)
```

`Criterion(metric=..., threshold=...)` is kept as ranking / target metadata, and
`max_rounds` is recorded for provenance, but they do not stop the PTB loop. If no
explicit wall-clock cap is given, the default is the PostTrainBench 10h budget.

## Run PostTrainBench Agent Mode

The real PostTrainBench path is:

```text
main-agent planner -> one exp sub-agent -> data operators -> Slurm train -> frozen eval
```

Machine-specific details are provided through `--runtime-config`, not through the playbook:

```bash
.venv/bin/python -m dataos_auto.cli run \
  --project ptb_gsm8k_run \
  --workspace workspace_ptb \
  --backend claude_code \
  --claude-code-model sonnet \
  --train-executor subagent \
  --runtime-config configs/h100.json \
  --target-bench gsm8k \
  --h100-base-model /data/hcgu/hf_cache/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/<snapshot> \
  --ptb-profile smoke \
  --max-wall-clock-s 3600 \
  --poll-interval-s 30
```

Use `--ptb-profile formal` for formal evaluation. `smoke` caps SFT data to
1000 rows and uses a small eval limit for end-to-end checks.

Resume: rerun the same `--project` and `--workspace`. Completed rounds
(`review.json` exists) are skipped. A sub-agent experiment resumes the same
Claude Code session until it writes a terminal `report.json`.

## Runtime Config

`configs/h100.json` is the current H100 config example. It contains paths and
key-file locations only, never secret values.

Expected shape:

```json
{
  "cluster": {
    "host": "114.214.215.159",
    "user": "hcgu",
    "port": 22,
    "key": "~/.ssh/ghc_miyao",
    "partition": "gpu",
    "qos": "gpu_priv",
    "gres": "gpu:1"
  },
  "paths": {
    "deploy_root": "/data/hcgu/users/tjh/dataos_agentver",
    "runs_base": "/data/hcgu/users/tjh/dataos_agentver_runs",
    "fixed_pool": "/data/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v1_full",
    "hf_cache": "/data/hcgu/hf_cache",
    "conda_sh": "/data/software/miniconda3/etc/profile.d/conda.sh",
    "train_env": "/data/hcgu/.conda/envs/train",
    "eval_env": "/data/hcgu/.conda/envs/lm-eval"
  },
  "gateway": {
    "base_url": "https://api2.starteam.wang/v1",
    "model": "claude-haiku-4-5",
    "key_file": "/data/hcgu/users/tjh/.gateway_key"
  }
}
```

Important fields:

| field | meaning |
| --- | --- |
| `paths.deploy_root` | deployed DataOS-AgentVer repo on the cluster |
| `paths.runs_base` | remote experiment artifact root |
| `paths.fixed_pool` | fixed PostTrainBench source pool |
| `paths.hf_cache` | Hugging Face model/cache root used offline by compute nodes |
| `paths.train_env` / `paths.eval_env` | conda envs for train/data and eval |
| `gateway.*` | login-node LLM-scoring endpoint and key-file path |

The main-agent injects this config into each sub-agent prompt as a
`RUNTIME CONFIG` block. `run-dataos/harnesses/posttrainbench/PLAYBOOK.md`
contains execution rules only; examples in the playbook must be parameterized
from `RUNTIME CONFIG`.

`--slurm` remains as a legacy inline JSON override, but new runs should use
`--runtime-config`.

## Verification

```bash
# L0/L1 local checks for the current loop and sub-agent executor
python -m pytest tests/test_subagent_exec.py tests/test_auto_loop.py

# L1 — real LLM produces a parseable PlanOut (auto-skips without key/CLI)
python -m pytest tests/test_auto_l1.py
```

Before finishing code changes, run targeted Ruff on changed files. Full Ruff
currently includes vendored/generated PostTrainBench assets and may report
pre-existing issues outside the runtime surface:

```bash
python -m ruff check dataos_auto tests
```

## Branch: agent-version-ptb

On the `agent-version-ptb` branch, eval is fixed to PostTrainBench. The planner
chooses data recipes and SFT hyperparameters; it does not change the scorer.

Execution roles:

| role | responsibility |
| --- | --- |
| main-agent | pick tree edges, launch exp sub-agents, resume sessions, attach reports, review results |
| exp sub-agent | implement exactly one experiment: build data, train, eval, write `report.json` |

Sub-agent constraints:

- Experiments run sequentially; PTB uses one H100 at a time.
- Non-LLM data processing, training, and eval generation run inside Slurm jobs.
- Login node work is limited to `sbatch`, `sacct`/`squeue`, and bounded internet-only steps.
- If an `llm_*` operator is used, only the bounded API scoring step may run on the login node.
  Ordinary operators, full-pool reads, final `save_to_disk`, train, and eval stay off the login node.
- Every experiment must write `report.json` with `status:"completed"` or `status:"failed"`.

The `report.json` path is local under:

```text
<workspace>/<project>/<study>/experiments/<exp>/report.json
```

Remote artifacts live under:

```text
<runtime_config.paths.runs_base>/<project>/<study>/<exp>/
```
