from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalConfig:
    """lm-evaluation-harness shaped eval configuration.

    Mirrors the subset of ``lm_eval.simple_evaluate`` we currently expose
    plus the DataOS adapter dispatch fields. Used directly by
    ``run_eval_stage`` (constructed via ``EvalConfig(**task.eval_config)``).
    """

    tasks: list[str]
    limit: int | None = None
    num_fewshot: int | None = None
    batch_size: int = 8
    max_new_tokens: int = 256
    apply_chat_template: bool = True
    confirm_run_unsafe_code: bool = False
    adapter: str = "transformers"
    adapter_config: dict = field(default_factory=dict)
