import torch
import pytest

from sakuramoti.flow_model.raft import RAFT

device = 'cuda'

@pytest.mark.parametrize("model_name", ["base", "small"])
def test_raft_config(model_name):
    conf = {"raft_model": model_name, "pretrained": None}
    model = RAFT(**conf)
    assert model.args.raft_model == model_name 
    
@pytest.mark.parametrize("model_name, pretrained", [("base", "sintel"), ("small", "small")])
@pytest.mark.parametrize("mixed_precision", [True, False])
def test_raft_smoke(model_name, pretrained, mixed_precision):
    conf = {"raft_model": model_name, "pretrained": pretrained ,"mixed_precision": mixed_precision}
    model = RAFT(**conf).to(device=device)
 
    image1 = torch.randn(32, 3, 256, 256, device=device)
    image2 = torch.randn(32, 3, 256, 256, device=device)
    output = model(image1, image2)
    assert output[0].shape == (32, 2, 256, 256)
    
"""
def test_alternate_corr_raft():
    # WIP Test Code
    # I want to make this code(https://github.com/princeton-vl/RAFT/tree/master/alt_cuda_corr) 
    # usable. However, since it's CUDA code,packaging becomes complicated. Therefore, I'm considering 
    # whether to separate it into another library.
    
    conf = {"alternate_corr": True}
    model = RAFT(**conf).to(device=device)
    image1 = torch.randn(32, 3, 256, 256, device=device)
    image2 = torch.randn(32, 3, 256, 256, device=device)
    output = model(image1, image2)
    assert output[0].shape == (32, 2, 256, 256)
"""