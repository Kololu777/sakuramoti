import torch
import torch.nn.functional as F
from torch import Tensor

from ..raft.corr import bilinear_sampler


class CorrBlock:
    def __init__(self, fmaps: Tensor, num_levels: int = 4, radius: int = 4):
        self.num_levels = num_levels
        self.radius = radius
        self.fmpas_pyramid = [fmaps]
        b, s, c, h, w = fmaps.shape
        for _ in range(self.num_levels - 1):
            _fmaps = fmaps.reshape(b * s, c, h, w)
            _fmaps = F.avg_pool2d(_fmaps, 2, stride=2)
            h, w = _fmaps.shape[2:]
            fmaps = _fmaps.reshape(b, s, c, h, w)
            self.fmpas_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        # WIP Code
        # b, s, n, d= SAKURAMOTI_TENSOR_CHECK(coords,
        #                                     'b s n d',
        #                                     'd=2',
        #                                     is_shape_return=True)
        b, s, n, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]
            _, _, _, h, w = corrs.shape
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dx, dy, indexing="ij"), dim=-1).to(coords.device)

            centroid_lvl = coords.reshape(b * s * n, 1, 1, 2) / 2**i
            delta_lvl = delta.reshape(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(corrs.reshape(b * s * n, 1, h, w), coords_lvl)
            corrs = corrs.view(b, s, n, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)
        return out.contiguous().float()

    def corr(self, targets):
        # WIP Code
        # b, s, n, c = SAKURAMOTI_TENSOR_CHECK(target,
        #                                      'b s n c',
        #                                      f'c=={self.c},
        #                                      f's=={self.s},
        #                                      is_shape_return=True
        # )
        b, s, n, c = targets.shape
        fmap1 = targets
        self.corrs_pyramid = []
        for fmaps in self.fmpas_pyramid:
            _, _, _, h, w = fmaps.shape
            fmap2s = fmaps.view(b, s, c, h * w)
            corrs = torch.matmul(fmap1, fmap2s)  # <(B, T, N, C), (B, T, C, H*W)> -> (B, T, M, H*W)
            corrs = corrs.view(b, s, n, h, w)
            corrs = corrs / torch.sqrt(torch.tensor(c).float())
            self.corrs_pyramid.append(corrs)

    def up_corr(self):
        b, s, n, h, w = self.corrs_pyramid[0].shape
        fcp = torch.zeros(
            (b, s, n, h, w),
            dtype=self.corrs_pyramid[0].dtype,
            device=self.corrs_pyramid[0].device,
        )
        for corr_level in range(self.num_levels):
            _fcp = self.corrs_pyramid[corr_level]
            _, _, _, _h, _w = _fcp.shape
            _fcp = _fcp.reshape(b * s, n, _h, _w)
            _fcp = F.interpolate(_fcp, (h, w), mode="bilinear", align_corners=True)
            fcp = fcp + _fcp.reshape(b, s, n, h, w)
        return fcp
