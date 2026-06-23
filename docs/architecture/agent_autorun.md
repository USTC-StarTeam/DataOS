# dataos.agent — autonomous SFT runtime (Agent SDK)

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

The user-facing entrypoint is one JSON file:

```bash
uv run dataos configs/runs/ptb-h100-qwen-gsm8k.json
```

The JSON contains run identity, workspace, goal, bench, base model path, time
budget, requested node/GPU/CPU resources, and fixed remote paths. Main-agent
strategy is not configured here; each round writes its own plan.

Resume: rerun the same JSON. Completed rounds
(`review.json` exists) are skipped. A sub-agent experiment resumes the same
Claude Code session until it writes a terminal `report.json`.

## Run Config

`configs/runs/ptb-h100-qwen-gsm8k.json` is the current H100/PTB run config
example. It contains no secret values.

Expected shape:

```json
{
  "run_id": "ptb-h100-qwen-gsm8k",
  "workspace": "workspace/ptb-h100-qwen-gsm8k",
  "goal": "Use the fixed PTB data pool to improve the base model within a limited H100 budget.",
  "bench": "gsm8k",
  "base_model_path": "/data/hcgu/hf_cache/Qwen/Qwen2.5-0.5B-Instruct",
  "time_budget_hours": 10,
  "resource": {
    "node": "gpu04",
    "gpus": 1,
    "cpus": 8
  },
  "paths": {
    "fixed_pool": "/data/hcgu/dataos_fixed_pools/processed/posttrainbench_fixed_pool_v1_full",
    "conda_sh": "/data/software/miniconda3/etc/profile.d/conda.sh",
    "train_env": "/data/hcgu/.conda/envs/train",
    "eval_env": "/data/hcgu/.conda/envs/lm-eval"
  }
}
```

Important fields:

| field | meaning |
| --- | --- |
| `run_id` | resumable project/run id |
| `workspace` | local run directory; it must end with `run_id` |
| `bench` | one PostTrainBench bench to optimize |
| `base_model_path` | exact H100 path of the base model |
| `resource.*` | requested node, GPU count, and CPU count |
| `paths.fixed_pool` | fixed PostTrainBench source pool |
| `paths.train_env` / `paths.eval_env` | conda envs for train/data and eval |

The CLI writes `run_config.json` and `resolved_config.json` into the workspace.
The resolved config expands internal defaults such as SSH, Slurm partition/qos,
remote deploy root, remote runs base, and key-file paths. The main-agent injects
that resolved runtime block into each sub-agent prompt.

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
python -m ruff check dataos.agent tests
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
