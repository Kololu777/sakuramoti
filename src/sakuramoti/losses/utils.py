from __future__ import annotations

import torch
from torch import Tensor


def maksed_mean(
    loss: Tensor,
    valid: Tensor | None = None,
    mean_mode: str = "mean",  # mean | "reduce_masked_mean" | TODO: "sum" | None
    eps: float = 1e-8,
) -> torch.Tensor:
    if valid is None:
        valid = torch.ones(*loss.shape)

    if mean_mode == "mean":
        return (valid * loss + eps).mean()

    elif mean_mode == "reduce_masked_mean":
        prod = loss * valid
        numer = torch.sum(prod)
        denom = torch.sum(valid) + eps
        return numer / denom

    else:
        mean_mode_error_message = {"mean", "reduce_masked_mean"}
        raise ValueError(f"The `mean_mode` must be one of {mean_mode_error_message}, but got {mean_mode}")
