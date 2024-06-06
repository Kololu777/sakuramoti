import torch
import torch.nn.functional as F
from torch import Tensor


def _coords_grid(batch: int, ht: int, wd: int, device: torch.device) -> Tensor:
    meshgrid = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device)
    )
    coords = torch.stack(meshgrid[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def _upflow8(flow: Tensor, mode="bilinear") -> Tensor:
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
