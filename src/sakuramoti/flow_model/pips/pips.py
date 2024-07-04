from __future__ import annotations
from typing import ClassVar
from types import SimpleNamespace
import torch.nn as nn
from .extractor import BasicEncoder
from .utils import bilinear_sample2d
from torch import Tensor
from .corr import CorrBlock
import torch
from .mlp_mixer import DeltaBlock

class Pips(nn.Module):
    default_conf: ClassVar[dict[str, any]] = {
        "model": "base",
        "pretrained": True
    }
    
    arch:ClassVar[dict[str, any]] = {
        "base": {
            "stride": 4,
            "hidden_dim": 256,
            "latent_dim": 128,
            "norm_fn": "instance",
            "corr_levels": 4,
            "corr_radius": 3,
            "fnet_dropout": 0.0,
            "s": 8
        }
    }
    
    __url = "https://huggingface.co/aharley/pips/resolve/main/model-000200000.pth"
    __pth_template = "pips_offcial.pth"
    
    def __init__(self, **conf_):
        super(Pips, self).__init__()
        self.args = args = SimpleNamespace(**{**self.default_conf, **conf_})
        for k, v in self.arch[args.model].items():
            setattr(args, k, v)
            
        self.fnet = BasicEncoder(output_dim=self.args.latent_dim,
                                 stride=self.args.stride,
                                 norm_fn=self.args.norm_fn,
                                 dropout=self.args.fnet_dropout)
    
        self.delta_block = DeltaBlock(input_dim=self.args.latent_dim, 
                                      hidden_dim=self.args.hidden_dim, 
                                      corr_levels=self.args.corr_levels, 
                                      corr_radius=self.args.corr_radius, 
                                      s=self.args.s)
        
        self.norm = nn.GroupNorm(1, self.args.latent_dim)
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.args.latent_dim, self.args.latent_dim),
            nn.GELU(),
        )
        
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.args.latent_dim, 1),
        )
        
        if self.args.pretrained is not None:
            state_dict = torch.hub.load_state_dict_from_url(url=self.__url, file_name=self.__pth_template)
            new_state_dict = {}
            for k, v in state_dict["model_state_dict"].items():
                if k in ["delta_block.to_delta.15.weight", "delta_block.to_delta.15.bias"]:
                    new_state_dict[k.replace("delta_block.to_delta.15.", "delta_block.to_delta.head.")] = v
                elif "delta_block.to_delta" in k:
                    new_state_dict[k.replace("delta_block.to_delta.", "delta_block.to_delta.mlp_mixer.")] = v
                else:
                    new_state_dict[k] = v
            self.load_state_dict(state_dict=new_state_dict)
    
    def init_pos_and_feat(self, 
                          xys:Tensor, 
                          fmaps:Tensor, 
                          coord_init:Tensor|None = None, 
                          feat_init:Tensor|None = None):
        _, n, _ = xys.shape
        b, s, _, _, _ = fmaps.shape
        # (3.3) Initialize poistions
        _xys = xys.clone() / float(self.args.stride) #Quantization
        if coord_init is None:
            # all pair
            coords = _xys.reshape(b, 1, n, 2).repeat(1, s, 1, 1) # B, N, 2 -> B, T, N, 2
        else:
            coords = coord_init.clone() /float(self.args.stride)
        
        # Blue Tile
        if feat_init is None:
            ffeats = bilinear_sample2d(fmaps[:, 0],
                                       coords[:, 0]).permute(0, 2, 1) # B, N, C
        else:
            ffeats = feat_init
        ffeats = ffeats.unsqueeze(1).repeat(1, s, 1, 1) #(B, T, N, C)
        return coords, ffeats
    
    
    def forward(self, 
                xys: Tensor,
                rgbs: Tensor,
                coord_init:Tensor | None = None,
                feat_init:Tensor | None = None,
                iters:int=3):
        _, n, _ = xys.shape #B, N, 2
        b, s, c, h, w = rgbs.shape #B, T, C, H, W (T=Frame)
        
        #(3.2) Extracting Features
        _rgbs = rgbs.reshape(b*s, c, h, w)
        _fmaps = self.fnet(_rgbs)
        fmaps = _fmaps.reshape(b, s, *_fmaps.shape[1:])
        
        # (3.3) Initialize poistions
        # coords: Shape of B, T, N, 2.
        # ffeats: Shape of B, T, N, C.
        coords, ffeats = self.init_pos_and_feat(xys, fmaps, coord_init=coord_init, feat_init=feat_init)
        

        fcorr_fn = CorrBlock(fmaps=fmaps, 
                             num_levels=self.args.corr_levels, 
                             radius=self.args.corr_radius)
        
        coord_predictions = [] # Quantization coordinate -> Input coordinate
        coord_predictions2 = [coords.detach() * self.args.stride, 
                              coords.detach() * self.args.stride] # Quantization coordinate -> Input coordinate

        fcps = []
        for _ in range(iters):
            # Measure local similarity
            coords = coords.detach() # "Remove from the computation graph."
            fcorr_fn.corr(ffeats)
            fcp = fcorr_fn.up_corr()
            fcps.append(fcp)
            fcorrs = fcorr_fn.sample(coords) # B, S, N, LRR
        
            # Update positions and features
            # for mixer, i want everything in the format B*N, S, C
            _ffeats:Tensor = ffeats.permute(0, 2, 1, 3).reshape(b*n, s, self.args.latent_dim)
            
            _, _, _, lrr = fcorrs.shape
            _fcorrs= fcorrs.permute(0, 2, 1, 3).reshape(b*n, s, lrr)
            
            _flows = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(b*n, s, 2)
            _times = torch.linspace(0, s, s, device=_flows.device).reshape(1, s, 1).repeat(b*n, 1, 1)
            _flows = torch.cat([_flows, _times], dim=2) # B*N, S, 3
            delta_all = self.delta_block(_ffeats, _fcorrs, _flows) # 8*(128 +2)
            
            # apply residual update
            _delta_coords = delta_all[:,:,:2]
            _delta_feats = delta_all[:,:,2:]
            
            # delta_{x}. Shape of B, N, S, 2.
            coords = coords + _delta_coords.reshape(b, n, s, 2).permute(0, 2, 1, 3)
            
            # delta_{F}. Shape of B, N, S, C.
            _ffeats = _ffeats.reshape(b*n*s, self.args.latent_dim)
            _delta_feats = _delta_feats.reshape(b*n*s, self.args.latent_dim)
            _ffeats = self.ffeat_updater(self.norm(_delta_feats)) + _ffeats
            ffeats = _ffeats.reshape(b, n, s, self.args.latent_dim).permute(0, 2, 1, 3)
            
            coord_predictions.append(coords * self.args.stride) # Quantization coordinate -> Input coordinate
            coord_predictions2.append(coords * self.args.stride) # Quantization coordinate -> Input coordinate
            
        vis_e = self.vis_predictor(ffeats.reshape(b*s*n, self.args.latent_dim)).reshape(b, s, n)
        
        coord_predictions2.append(coords * self.args.stride) # Quantization coordinate -> Input coordinate
        coord_predictions2.append(coords * self.args.stride) # Quantization coordinate -> Input coordinate
        
        return coord_predictions, coord_predictions2, vis_e