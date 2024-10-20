import torch
from torch import Tensor


def generate_grid(
    height: int, width: int, device: torch.device, normalize: bool = False, homogeneous: bool = False
) -> Tensor:
    if normalize:
        lin_x = torch.linspace(-1, 1, steps=width, device=device)
        lin_y = torch.linspace(-1, 1, steps=height, device=device)
    else:
        lin_x = torch.linspace(0, width, steps=width, device=device)
        lin_y = torch.linspace(0, height, steps=height, device=device)
    grid_x, grid_y = torch.meshgrid((lin_x, lin_y), indexing="xy")
    grid = torch.stack([grid_x, grid_y], dim=-1)

    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid


def normalize_coords(coords: Tensor, h: int, w: int, no_shift: bool = False) -> Tensor:
    """
    Normalize coordinates to be in the range of [-1, 1].
    coords: Shape of (w, h, 2). values are in the range of [0, w-1] or [0, h-1].

    Args:
        coords: Shape of (W, H, 2). 2D coordinate.
        h: height of image size.
        w: width of image size.
        no_shift: if True, do not shift the coordinate to [0, 2], False, shift to [-1, 1].
    """
    assert coords.shape[-1] == 2, f"shape of coords must be (*, 2), but got {coords.shape}"
    if no_shift:
        return coords / torch.tensor([w, h], device=coords.device) * 2  # May be wrong. Should be remove *2.
        # return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2
    else:
        return coords / torch.tensor([w, h], device=coords.device) * 2 - 1.0
        # return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0
