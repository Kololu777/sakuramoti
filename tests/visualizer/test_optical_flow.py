import torch
from sakuramoti.visualizer.optical_flow import make_colorwheel


def test_make_colorwheel(device, dtype):
    colorwheel = make_colorwheel()
    assert colorwheel.shape == torch.Size([3, 55])
