from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .utils import maksed_mean


def sequence_loss(
    inputs: list[Tensor],
    target: Tensor,
    valid: Tensor | None = None,
    gamma: float = 0.8,
    mean_mode: str = "mean",
):
    r"""Criterion that compute sequence loss [1][2] similar to l1 loss.

    Reference:
        [1] RAFT: https://arxiv.org/pdf/2003.12039
        [2] PIPs: https://arxiv.org/pdf/2204.04153

    Parameters:
        input: The predicted tensor. Shape of (B, T, N, 2).
        target: The target tensor. Shape of (B, T, N, 2).
        valid: Shape of (B, N, 2). If `valid = None`, `valid = torch.ones(B, N, 2)`
        gamma: Exponentially increasing weights.
        mean_mode:
            - ``'mean'`` [default]: RAFT Sequence loss: Calculating the average the average from tensor.
            - ``'reduce_masked_mean'`` PIPs Sequence Loss: Calculating the average excluding 0 from the valid values.

    Returns:
        one-element tensor of the computed loss.

    Example:
        >>> T = 8
        >>> input = torch.rand(32, T, 100, 2) #(B, T, N, 2)
        >>> inputs = [input for _ in range(3)] # iteration
        >>> output = sequence_loss(inputs, target)
        >>> output.backward()

    """
    n = len(inputs)
    flow_loss = 0.0
    for idx in range(n):
        weight = gamma ** (n - (idx + 1))
        loss = torch.mean((inputs[idx] - target).abs(), dim=3)
        flow_loss += weight * maksed_mean(loss, valid, mean_mode)
    return flow_loss


class SequenceLoss(nn.Module):
    r"""Criterion that compute Sequence Loss.

    Reference:
        [1] RAFT: https://arxiv.org/pdf/2003.12039
        [2] PIPs: https://arxiv.org/pdf/2204.04153

    Parameters:
        gamma: Exponentially increasing weights.
        mean_mode:
            - ``'mean'`` [default]: RAFT Sequence loss: Calculating the average the average from tensor.
            - ``'reduce_masked_mean'`` PIPs Sequence Loss: Calculating the average excluding 0 from the valid values.

    Shape:
       - input: :math:`(B, T, N, 2)` where T = number of frame, N = number of keypoints
       - target: :math:`(B, T, N, 2)`.
       - valid: :math:`(B, T, N)` where type of vaild is torch.bool.

    Example:
        >>> T = 8
        >>> input = torch.rand(32, T, 100, 2) #(B, T, N, 2)
        >>> inputs = [input for _ in range(3)] # iteration
        >>> target = torch.rand(32, T, 100, 2) #(B, T, N, 2)
        >>> valid = torch.randint(0, 2, (32, T, 100), dtype=torch.bool)
        >>> criterion = SequenceLoss()
        >>> output = criterion(inputs, target, vai)
        >>> output.backward()

    """

    def __init__(self, gamma: float = 0.8, mean_mode: str = "mean"):
        super().__init__()
        self._gamma = gamma
        self._mean_mode = mean_mode

    def forward(
        self, inputs: list[Tensor], target: Tensor, valid: Tensor | None = None
    ) -> Tensor:
        return sequence_loss(
            inputs=inputs,
            target=target,
            valid=valid,
            gamma=self._gamma,
            mean_mode=self._mean_mode,
        )
