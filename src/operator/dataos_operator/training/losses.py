"""Loss helpers shared by trainer integrations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_per_sample_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return one causal-language-modeling loss value per sample."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(shift_labels.size(0), -1)
    valid_mask = shift_labels.ne(ignore_index)
    valid_counts = valid_mask.sum(dim=1).clamp(min=1)
    return (token_losses * valid_mask).sum(dim=1) / valid_counts
