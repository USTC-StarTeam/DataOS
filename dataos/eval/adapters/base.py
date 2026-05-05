from __future__ import annotations

from typing import Protocol, runtime_checkable

from dataos.core.artifact import ModelArtifact
from dataos.eval.types import AdapterCallContext, EvalRequest, ModelResponse


class EvalModelAdapter(Protocol):
    name: str

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: AdapterCallContext,
    ) -> list[ModelResponse]: ...


@runtime_checkable
class EvalLoglikelihoodAdapter(Protocol):
    """Adapters that can score arbitrary continuations against a context.

    Used by ``DataOSLM.loglikelihood`` so lm-eval ``multiple_choice`` /
    ``loglikelihood`` tasks (mmlu, arc, hellaswag, truthfulqa, ...) can run
    through the bridge. Capability-orthogonal to :class:`EvalModelAdapter`:
    an adapter may implement either, both, or neither. ``isinstance(adapter,
    EvalLoglikelihoodAdapter)`` is the runtime feature check used by the
    planner / bridge.
    """

    name: str

    def score_continuations(
        self,
        model_artifact: ModelArtifact,
        pairs: list[tuple[str, str]],
        config: AdapterCallContext,
    ) -> list[tuple[float, bool]]:
        """Score ``(context, continuation)`` pairs.

        Returns one ``(continuation_token_logprob_sum, is_greedy)`` tuple per
        input pair. ``is_greedy`` is True iff each continuation token is
        argmax of the model's distribution at that position given the
        preceding tokens (i.e. greedy decoding from ``context`` would have
        produced exactly this ``continuation``).
        """
        ...
