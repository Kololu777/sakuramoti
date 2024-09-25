# BSD 3-Clause License

# Copyright (c) 2020, princeton-vl
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Authors:
# Zachary Teed, Jia Deng
# Code from https://github.com/princeton-vl/RAFT

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sakuramoti.utils.hub import load_state_dict_from_zip_url
from sakuramoti.flow_model.raft.corr import CorrBlock, AlternateCorrBlock
from sakuramoti.flow_model.raft.utils import _upflow8, _coords_grid
from sakuramoti.flow_model.raft.update import (
    BasicUpdateBlock,
    SmallUpdateBlock,
)
from sakuramoti.flow_model.raft.extractor import BasicEncoder, SmallEncoder

try:
    autocast = torch.cuda.amp.autocast
except:  # noqa: E722
    # dummy autocast for PyTorch < 1.6
    class autocast:  # type: ignore [no-redef]
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    default_conf: ClassVar[dict[str, any]] = {
        "raft_model": "base",  # small or base
        "dropout": 0.0,
        "alternate_corr": False,
        "pretrained": "sintel",  # chairs, sintel, things, kitti, small, None.
        "mixed_precision": False,
    }

    arch: ClassVar[dict[str, any]] = {
        "small": {
            "hidden_dim": 96,
            "context_dim": 64,
            "encoder_feature_dim": 128,
            "corr_levels": 4,
            "corr_radius": 3,
        },
        "base": {
            "hidden_dim": 128,
            "context_dim": 128,
            "encoder_feature_dim": 256,
            "corr_levels": 4,
            "corr_radius": 4,
        },
    }

    # https://github.com/princeton-vl/RAFT/blob/master/download_models.sh
    __url = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
    __pth_template = "raft-{}.pth"

    def __init__(self, **conf_):
        super().__init__()
        self.args = args = SimpleNamespace(**{**self.default_conf, **conf_})
        for k, v in self.arch[args.raft_model].items():
            setattr(args, k, v)

        # feature network, context network, and update block
        if args.raft_model == "small":
            self.fnet = SmallEncoder(output_dim=128, norm_fn="instance", dropout=args.dropout)
            self.cnet = SmallEncoder(
                output_dim=args.hidden_dim + args.context_dim,
                norm_fn="none",
                dropout=args.dropout,
            )
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=args.hidden_dim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=args.dropout)
            self.cnet = BasicEncoder(
                output_dim=args.hidden_dim + args.context_dim,
                norm_fn="batch",
                dropout=args.dropout,
            )
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=self.args.hidden_dim)

        if self.args.pretrained is not None:
            state_dict = load_state_dict_from_zip_url(
                url=self.__url,
                target_file_name=self.__pth_template.format(self.args.pretrained),
            )
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            self.load_state_dict(state_dict=new_state_dict)

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img: Tensor) -> tuple[Tensor, Tensor]:
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = _coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = _coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow: Tensor, mask: Tensor) -> Tensor:
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, (3, 3), padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def _run(
        self,
        image1: Tensor,
        image2: Tensor,
        iters: int = 12,
        flow_init: int | None = None,
        test_mode: bool = False,
    ):
        """Estimate optical flow between pair of frames"""

        hdim = self.args.hidden_dim
        cdim = self.args.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = _upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions

    def forward(
        self,
        image1: Tensor,
        image2: Tensor,
        iters: int = 12,
        flow_init: int | None = None,
    ) -> list[Tensor]:
        """Estimate optical flow between pair of frames. (train)"""
        return self._run(image1, image2, iters, flow_init, test_mode=False)

    @torch.no_grad()
    def pred(
        self,
        image1: Tensor,
        image2: Tensor,
        iters: int = 12,
        flow_init: int | None = None,
    ) -> tuple(Tensor, Tensor):
        return self._run(image1, image2, iters, flow_init, test_mode=True)
