from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LEGACY_ADAPTER_MODES = {"legacy", "legacy_adapter", "dataos_adapter"}
NATIVE_LM_EVAL_MODES = {"lm_eval_native", "native"}


@dataclass(slots=True)
class EvalConfig:
    """lm-evaluation-harness shaped eval configuration.

    ``mode="lm_eval_native"`` (the default and only mode) delegates model loading and
    inference to lm-evaluation-harness backends such as ``hf`` / ``vllm`` /
    ``local-chat-completions`` (plus the DataOS tool-chat plugin in
    :mod:`dataos.integrations.lm_eval.models`). The pre-0.2.5 ``legacy_adapter`` bridge was removed:
    endpoint-served models go through lm-eval's own API backends, and anything lm-eval
    can't score goes through ``provider="external"`` instead.
    """

    tasks: list[str]
    mode: str = "lm_eval_native"
    limit: int | float | None = None
    num_fewshot: int | None = None
    batch_size: int | str | None = 8
    max_batch_size: int | None = None
    max_new_tokens: int = 256
    apply_chat_template: bool | str = True
    fewshot_as_multiturn: bool = False
    system_instruction: str | None = None
    gen_kwargs: dict[str, Any] | str | None = None
    samples: dict[str, list[int]] | str | None = None
    confirm_run_unsafe_code: bool = False
    bootstrap_iters: int = 0
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    predict_only: bool = False
    device: str | None = None
    use_cache: str | None = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    seed: int | list[int | None] | tuple[int | None, ...] | None = None
    random_seed: int | None = 0
    numpy_random_seed: int | None = 1234
    torch_random_seed: int | None = 1234
    fewshot_random_seed: int | None = 1234
    model: str | None = None
    model_args: dict[str, Any] | str | None = None
    include_path: str | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool | None = None
    output_path: str | None = None
    headline_metric: str | None = None

    def normalized_mode(self) -> str:
        mode = self.mode.strip().lower()
        if mode in NATIVE_LM_EVAL_MODES:
            return "lm_eval_native"
        if mode in LEGACY_ADAPTER_MODES:
            raise ValueError(
                "the legacy_adapter eval bridge was removed in 0.2.5 — drop the 'mode' key "
                "(lm_eval_native is the default; endpoints go through lm-eval's API backends "
                "such as local-chat-completions), or use provider='external' for benches "
                "lm-eval can't score"
            )
        raise ValueError(
            f"Unknown eval mode {self.mode!r}; expected one of: {sorted(NATIVE_LM_EVAL_MODES)}"
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
