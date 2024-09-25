from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .utils import maksed_mean


def balanced_cross_entropy_loss_with_logits(
    input: Tensor,
    target: Tensor,
    valid: Tensor | None = None,
    max_th: float = 0.95,
    min_th: float = 0.05,
) -> torch.Tensor:
    """
    Criterion that compute balanced cross entropy

    Paramaters:
        inputs: The predicted tensor. Shape of (B, C, *)
        target: The target tensor. Shape of (B, C, *)
        valid: The mask tensor. Shape of (B, C, *). If `valid = None`, `valid = torch.ones(B, C, *)`
        max_th: Caluculate only the `target` values greather than `max_th` as `pos_loss`
        min_th: Caluculate only the `target` values lower than `min_th` as `neg_loss`

    """
    if valid is None:
        valid = torch.ones_like(input)
    pos_valid = (target > max_th).float() * valid
    neg_valid = (target < min_th).float() * valid

    log_probs_pos = nn.functional.logsigmoid(input)  # log(sig(x))
    log_probs_neg = nn.functional.logsigmoid(-input)  # 1 - log(sig(x))

    pos_term = -log_probs_pos * target
    neg_term = -log_probs_neg * (1.0 - target)

    pos_loss = maksed_mean(loss=pos_term, valid=pos_valid, mean_mode="reduce_masked_mean")
    neg_loss = maksed_mean(loss=neg_term, valid=neg_valid, mean_mode="reduce_masked_mean")
    return pos_loss + neg_loss


class BalancedCrossEntropy(nn.Module):
    def __init__(
        self,
        max_th: float = 0.95,
        min_th: float = 0.05,
    ):
        super().__init__()
        self._max_th = max_th
        self._min_th = min_th

    def forward(self, input: Tensor, target: Tensor, valid: Tensor | None = None):
        return balanced_cross_entropy_loss_with_logits(input, target, valid, max_th=self._max_th, min_th=self._min_th)
