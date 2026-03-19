`dataos/` is the standalone package for the Agentic DataOS SFT MVP.

Phase 0 freezes the package boundary between `core`, `registry`, `operators`,
`planner`, `controller`, `runtime`, `storage`, and `eval` so later tasks can
fill in concrete behavior without blurring responsibilities.

The package already contains:

- stable shared schemas in `core/`
- concrete or placeholder interfaces in `registry/`, `planner/`, `controller/`,
  `runtime/`, `storage/`, and `eval/`
- operator namespaces in `operators/`

The intent is to keep orchestration decisions in `controller`, plan generation
in `planner`, and execution concerns in `runtime`, instead of letting those
roles drift together as the implementation grows.

The T0 baseline additionally freezes three shared contracts:

- built-in runtime operators use bare `snake_case` IDs such as `filter_by_rule`
  and `sample_by_strategy`
- `simple_sft_eval` is an evaluator in `eval/`, not a runtime operator
- `PlanningResult.pipeline` contains only runtime steps; evaluation happens
  after runtime and is orchestrated by `controller`
