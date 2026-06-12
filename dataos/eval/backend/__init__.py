"""lm-evaluation-harness backend for DataOS eval (native mode only)."""

from dataos.eval.backend.native import run_lm_eval_native
from dataos.eval.types import LmEvalRunOutput

__all__ = ["LmEvalRunOutput", "run_lm_eval_native"]
