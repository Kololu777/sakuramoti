from __future__ import annotations

import torch
from torch import Tensor


def comupte_n_px(epe: Tensor, n: Tensor) -> Tensor:
    return (epe < n).float().mean().item()


end_point_error_template = {"epe": None, "1px": 1, "3px": 3, "5px": 5}


def end_point_error(
    pred: Tensor,
    target: Tensor,
    valid: Tensor | None = None,
    p: int = 2,
    npx: list[int] = [1, 3, 5],
) -> dict[Tensor]:
    """compute end point error.

    Args:
        pred: The prediction tensor. Shape of (B, T, ...).
        target: The target tensor (Ground Truth). Shape of (B, T, ...).
        valid: The mask tensor. Shape of (B, ...)
        p: the order of norm.
        npx: List of thresholds to compute metrics for end-point-error.

    Returns:
        dict[Tensor]: end point error.
    """

    if valid is not None:
        assert (
            pred.shape[0] == valid.shape[0] and pred.shape[2:] == valid.shape[1:]
        ), f"Shape of `pred` and `valid` are not paired. Expected to shape of `pred`, `valid` is (B, T, ...), (B, ...) respectivly. but Get to shape of `pred`, `valid` is {pred.size()}, {valid.size()}."
    epe = torch.norm(pred - target, p=p, dim=1)
    if valid is not None:
        epe[valid]

    metrics = {}
    metrics["epe"] = epe.mean().item()

    for n in npx:
        metrics[f"{str(n)}px"] = comupte_n_px(epe, n)
    return metrics
