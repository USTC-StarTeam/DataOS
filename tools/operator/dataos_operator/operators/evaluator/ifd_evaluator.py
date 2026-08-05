"""使用语言模型计算样本 instruction following difficulty 分数的 evaluator。"""

from __future__ import annotations

from typing import Any

from dataos_operator.core import Record
from dataos_operator.operators.evaluator.perplexity_evaluator import PerplexityEvaluator


class IFDEvaluator(PerplexityEvaluator):
    """Compute the Instruction-Following Difficulty (IFD) score: response loss with vs. without the instruction.

    Implementation: reuse PerplexityEvaluator's context building to compute the "response loss
    with query" and the "response-only loss" separately, write their ratio to ``score_key``
    (default ``metadata.ifd``), and attach both losses for analysis; reuse an existing score
    when ``overwrite=False``.
    Dependency: local-model operator; loads a HF causal LM on demand (transformers/torch).
    """

    def __init__(
        self,
        hf_model: str,
        model_params: dict[str, Any] | None = None,
        score_key: str = "metadata.ifd",
        query_template: str | None = None,
        response_template: str | None = None,
        messages_key: str = "messages",
        system_key: str = "system",
        text_key: str = "text",
        query_key: str = "query",
        answer_key: str = "answer",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            hf_model=hf_model,
            model_params=model_params,
            score_key=score_key,
            query_template=query_template,
            response_template=response_template,
            messages_key=messages_key,
            system_key=system_key,
            text_key=text_key,
            query_key=query_key,
            answer_key=answer_key,
            overwrite=overwrite,
            **kwargs,
        )
        self.loss_with_query_key = f"{score_key}_loss_with_query"
        self.loss_without_query_key = f"{score_key}_loss_without_query"

    def evaluate_record(self, record: Record) -> Record:
        if not self.overwrite and self.get_field(record, self.score_key, None) is not None:
            return dict(record)

        prompt_text, response_text = self._prompt_and_response(record)
        loss_with_query = self._compute_loss(prompt_text, response_text)
        loss_without_query = self._compute_loss("Assistant: ", response_text)
        if loss_without_query == 0.0:
            raise ValueError("loss_without_query must not be zero when computing IFD")

        scored_record = self.set_field(record, self.score_key, loss_with_query / loss_without_query)
        scored_record = self.set_field(scored_record, self.loss_with_query_key, loss_with_query)
        scored_record = self.set_field(
            scored_record,
            self.loss_without_query_key,
            loss_without_query,
        )
        return scored_record

    def _compute_loss(self, prompt_text: str, response_text: str) -> float:
        if not response_text:
            raise ValueError("response text must not be empty when computing IFD")

        self._ensure_model_loaded()
        assert self._model is not None
        assert self._torch is not None

        input_ids, attention_mask, prompt_length = self._encode_prompt_and_response(
            prompt_text,
            response_text,
        )

        labels = input_ids.clone()
        labels[:, :prompt_length] = -100

        device = next(self._model.parameters()).device
        with self._torch.no_grad():
            outputs = self._model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
        return float(outputs.loss.detach().cpu().item())
