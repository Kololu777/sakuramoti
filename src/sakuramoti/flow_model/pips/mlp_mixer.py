import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import partial

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

class MLPMixer(nn.Module):
    def __init__(
                self,
                s:int,
                input_dim:int,
                hidden_dim:int,
                output_dim:int,
                depth:int,
                expansion_factor:int=4,
                dropout:float=0.
                ):
        super(MLPMixer, self).__init__()
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.mlp_mixer = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        *[nn.Sequential(
            PreNormResidual(hidden_dim, FeedForward(s, expansion_factor, dropout, chan_first)),
            PreNormResidual(hidden_dim, FeedForward(hidden_dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim),
        self.head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x:Tensor):
        x = self.mlp_mixer(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

def get_3d_embedding(xyz, c, cat_coords=True):
    b, n, d = xyz.shape
    assert(d==3)

    x = xyz[:,:,0:1]
    y = xyz[:,:,1:2]
    z = xyz[:,:,2:3]
    div_term = (torch.arange(0, c, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / c)).reshape(1, 1, int(c/2))
    
    pe_x = torch.zeros(b, n, c, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(b, n, c, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(b, n, c, device=xyz.device, dtype=torch.float32)
    
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    
    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)
    
    pe = torch.cat([pe_x, pe_y, pe_z], dim=2) # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2) # B, N, C*3+3
    return pe    

class DeltaBlock(nn.Module):
    def __init__(self,
                 input_dim:int=128,
                 hidden_dim:int=128,
                 corr_levels:int=4,
                 corr_radius:int=3,
                 s:int=8):
        super(DeltaBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.s = s
        # Correlation Matrix: (corr_levels * (2 * corr_radius + 1) ** 2)
        # Features: self.input_dim
        # Positional Encoding: 64 * 3
        # Flows(2) and Times(1): 3
        kitchen_dim = (corr_levels * (2 * corr_radius + 1) ** 2) + self.input_dim + 64 * 3 + 3
        
        self.to_delta = MLPMixer(
            s=self.s,
            input_dim=kitchen_dim,
            hidden_dim=512,
            output_dim=self.s * (input_dim + 2),
            depth=12
        )
    
    def forward(self, fhid:Tensor, fcorr:Tensor, flow:Tensor)->Tensor:
        b, _, _ = flow.shape
        flow_sincos = get_3d_embedding(flow, 64, cat_coords=True)
        x = torch.cat([fhid, fcorr, flow_sincos], dim=2) #B, S, kitchecn_dim
        delta = self.to_delta(x)
        delta = delta.reshape(b, self.s, self.input_dim+2) #B, S, self.input_dim + 2
        return delta