"""Run the frozen VLMEvalKit entrypoint with the Qwen TP communication workaround."""

from __future__ import annotations

import runpy
import sys


def configure_nccl() -> None:
    import vllm

    # Preserve tensor parallelism, model and decoding settings; use NCCL instead of the
    # custom CUDA-IPC all-reduce kernel which fails during graph setup on gpu01.
    original = vllm.LLM

    def create_llm(*args, **kwargs):
        kwargs["disable_custom_all_reduce"] = True
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            # Upstream catches Exception per benchmark and otherwise starts the same broken
            # engine eight times. Engine initialization failure invalidates the whole job.
            raise SystemExit(f"vLLM initialization failed: {exc}") from exc

    vllm.LLM = create_llm


def main() -> None:
    configure_nccl()
    script = sys.argv.pop(1)
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
