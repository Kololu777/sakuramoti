import torch
import pytest
from sakuramoti.flow_model.pips.mlp_mixer import MLPMixer, DeltaBlock


@pytest.mark.parametrize("input_tensor_shape", [(32, 12, 128), (1, 12, 128)])
def test_mlp_mixer_smoke(input_tensor_shape):
    model = MLPMixer(s=12, input_dim=128, hidden_dim=256, output_dim=10, depth=8, expansion_factor=4, dropout=0.0)
    input = torch.rand(input_tensor_shape)  # B, N, C
    output = model(input)
    assert output.shape == (input_tensor_shape[0], 10)


def test_delta_block():
    model = DeltaBlock()
    fhid = torch.rand(32, 8, 4 * (2 * 3 + 1) ** 2)
    fcorr = torch.rand(32, 8, 128)
    flow = torch.rand(32, 8, 3)
    x = model(fhid, fcorr, flow)
    assert x.shape == (32, 8, 130)
