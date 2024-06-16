import torch
import torch.nn as nn
from torch import Tensor

def maksed_mean(loss, valid, mean_mode, eps=1e-8):
    if mean_mode == "mean":
        return (valid * loss).mean()
    
    elif mean_mode == "reduce_masked_mean":
        prod = loss * valid
        numer = torch.sum(prod)
        denom = eps+torch.sum(valid)
        return numer / denom



def sequence_loss(input:list[Tensor], 
                  target:Tensor, 
                  valids:Tensor,
                  gamma:float=0.8, 
                  mean_mode:str='reduce_masked_mean'):
    """Criterion that compute sequence loss [1][2] similar to l1 loss.

    Reference:
        [1] RAFT: https://arxiv.org/pdf/2003.12039
        [2] PIPs: https://arxiv.org/pdf/2204.04153

    Args:
        input (list[Tensor]): The predicted tensor. Shape of ()
        target (Tensor): The target tensor. Shape of ()
        valids (Tensor): 
        gamma (float, optional): exponentially increasing weights.
        mean_mode (str, optional):
            - ``'reduce_masked_mean'``: PIPs Sequence Loss: Calculating the average excluding 0 from the valid values.
            - ``'mean'``: RAFT Sequence loss: Calculating the average the average from tensor.

    Returns:
        _type_: _description_
    """
    n = len(input)
    flow_loss = 0.0    
    for idx in range(1, n + 1, 1): 
        weight =  gamma ** (n - idx)
        loss = torch.mean((input[idx] - target).abs(), dim=3)
        flow_loss += weight * maksed_mean(loss, valids)
    return flow_loss
            
class SequenceLoss(nn.Module):
    def __init__(self, gamma:float=0.8, max_flow:int=400):
        self._gamma = gamma
        self._max_flow = max_flow
        
    def forward(self, input:list[Tensor], target:Tensor, valids:Tensor)->Tensor:
        return sequence_loss(input=input, 
                             target=target, 
                             valids=valids,
                             gamma=self._gamma, 
                             max_flow=self._max_flow)