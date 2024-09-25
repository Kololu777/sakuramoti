import torch
from sakuramoti.flow_model.pips.pips import Pips


def test_pips_smoke():
    xys = torch.randint(low=0, high=2456, size=(4, 100, 2)).to("cuda")
    rgbs = torch.rand(4, 8, 3, 256, 256).to("cuda")  # 4Frame
    model = Pips().to("cuda")
    coord_pred, coord_pred2, vis_e = model(xys, rgbs)
    assert coord_pred[0].shape == (4, 8, 100, 2)  # B, T, N, 2
    assert len(coord_pred) == 3
    assert vis_e.shape == (4, 8, 100)
    assert coord_pred2[0].shape == (4, 8, 100, 2)
    assert len(coord_pred2) == 7
