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
  project.set_goal(criterion, budget)   # persisted to project.json (resume reads it back)
  proj.load()                           # rehydrate prior studies (resume)
  freeze_eval() once  →  prepare_once() shared data
  for each round (deterministic):
    stop_reason()?  → criterion / budget:rounds / budget:wall_clock   # LLM never stops the loop
    planner.next_plan()   ← LLM judgment point #1 (Agent.run(output=PlanOut))
    study.set_plan()  (eval overwritten with the frozen config)
    executor.fan_out()    → one run_one.py subprocess per experiment, GPU-pool concurrency,
                            report.json polled as authoritative status, skip_completed
    reviewer.conclude()   ← LLM judgment point #2 (Agent.run(output=ReviewOut))
    study.review()
```

## Stop conditions (exactly three, first to fire wins)

| reason | config |
| --- | --- |
| `criterion` — bench score target | `Criterion(metric="score"|<bench>, threshold=…)` |
| `budget:rounds` — max rounds | `Budget(max_rounds=…)` |
| `budget:wall_clock` — max seconds | `Budget(max_wall_clock_s=…)` |

At least one budget cap (or a target) is required — no unbounded auto-run.

## Run

Real (a800, needs `MIMO_API_KEY` in env + `claude` CLI on PATH):

```bash
dataos-auto run --project ptb --backend claude \
  --goal "raise PostTrainBench native benches" \
  --eval-cfg '{"tasks":["gsm8k","bfcl","humaneval"]}' \
  --data-source hf://<repo> --gpus 4,5,6 \
  --target 0.85 --max-rounds 5 --max-wall-clock-s 5400
```

Resume: rerun the same `--project` — completed rounds (studies with a `review.json`) are
skipped, unfinished experiments rerun via the library's completed-skip + liveness guards.

## Verification

```bash
# L0 — full skeleton, no GPU / no LLM / no claude CLI
python -m pytest tests/test_auto_budget.py tests/test_auto_executor.py tests/test_auto_loop.py
# L1 — real LLM produces a parseable PlanOut (auto-skips without key/CLI)
python -m pytest tests/test_auto_l1.py
```

## Notes

- Keys come from env only (`MIMO_API_KEY`/`ANTHROPIC_API_KEY`), never config or disk.
- The `stub` backend (used by L0 tests) needs fixed-JSON replies via `cfg.stub_replies`;
  the CLI is for the `claude` backend.
- Eval auto-discovery is deferred — supply the frozen eval via `--eval-cfg`.
- Remote `SSHRunner` is a reserved seam; the MVP fans out locally.

## Branch: agent-version-ptb (fixed PostTrainBench eval)

On the `agent-version-ptb` branch the evaluation is **not configurable** — `freeze_eval`
ignores `eval_cfg` and always returns the frozen PostTrainBench suite ([ptb.py](ptb.py)):
the 7 official benches via the external harness (`run_ptb.sh` → official `evaluate.py`),
macro-averaged. `--ptb-profile formal|smoke` picks limits (default `formal`). Benches whose
key/token is absent (`~/.mimo_key` for healthbench/arenahard judge, `HF_TOKEN` for gpqamain)
fail in the harness and are skipped by the suite combiner. Running requires the official
scorer deployed and pointed at by `PTB_ROOT` (vllm 0.11.0); `--stub` still works for L0.
