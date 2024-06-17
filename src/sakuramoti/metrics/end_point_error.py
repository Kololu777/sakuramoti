from __future__ import annotations
import torch
from torch import Tensor

def end_point_error(pred:Tensor, target:Tensor, valid:Tensor | None =None)->dict[Tensor]:
    epe = torch.sum((pred - target) ** 2, dim=1).sqrt()
    if valid is not None:
        epe = epe.view(-1)[valid.view(-1)]
    
    metrics = {"epe": epe.mean().item(),
               "1px": (epe < 1).float().mean().item(),
               "3px": (epe < 3).float().mean().item(),
               "5px": (epe < 5).float().mean().item()
    }
    return metrics 