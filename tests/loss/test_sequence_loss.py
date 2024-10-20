import torch
import pytest
from sakuramoti.losses.sequence_loss import SequenceLoss, sequence_loss

from tests.base import BaseTester


class TestSequenceLoss(BaseTester):
    @pytest.mark.parametrize("mean_mode", ["reduce_masked_mean", "mean"])
    def test_smoke(self, device, dtype, mean_mode):
        input = [torch.rand((32, 8, 100, 2), device=device, dtype=dtype) for _ in range(2)]
        valid = torch.randint(0, 2, (32, 8, 100), device=device, dtype=torch.bool)
        target = torch.rand((32, 8, 100, 2), device=device, dtype=dtype)
        sl = SequenceLoss(mean_mode=mean_mode)
        assert isinstance(sl(input, target, valid), torch.Tensor)

    def test_exception(self, device, dtype):
        input = [torch.rand((32, 8, 100, 2), device=device, dtype=dtype) for _ in range(2)]
        valid = torch.randint(0, 2, (32, 8, 100), device=device, dtype=torch.bool)
        target = torch.rand((32, 8, 100, 2), device=device, dtype=dtype)

        with pytest.raises(ValueError) as errinfo:
            SequenceLoss(mean_mode="not_mean_mode")(input, target, valid)
        mean_mode_error_message = {"mean", "reduce_masked_mean"}
        assert f"The `mean_mode` must be one of {mean_mode_error_message}, but got not_mean_mode" in str(errinfo)

    def test_gradcheck(self, device):
        input = [torch.rand((32, 8, 100, 2), device=device, dtype=torch.float64) for _ in range(2)]
        valid = torch.randint(0, 2, (32, 8, 100), device=device, dtype=torch.bool)
        target = torch.rand((32, 8, 100, 2), device=device, dtype=torch.float64)
        self.gradcheck(sequence_loss, (input, target, valid), dtypes=[torch.float64, torch.float64, torch.bool])

    def test_module(self, device, dtype):
        input = [torch.rand((32, 8, 100, 2), device=device, dtype=dtype) for _ in range(2)]
        valid = torch.randint(0, 2, (32, 8, 100), device=device, dtype=torch.bool)
        target = torch.rand((32, 8, 100, 2), device=device, dtype=dtype)
        op = sequence_loss
        op_module = SequenceLoss()
        self.assert_close(op(input, target, valid), op_module(input, target, valid))
