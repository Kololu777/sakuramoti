import torch
import pytest
import torch.nn as nn
from sakuramoti.losses.balanced_ce_loss import BalancedCrossEntropy, balanced_cross_entropy_loss_with_logits

from tests.base import BaseTester

logsig_05 = -nn.functional.logsigmoid(torch.tensor(0.5))
logsig_m05 = -nn.functional.logsigmoid(torch.tensor(-0.5))
logsig_05_m05 = -nn.functional.logsigmoid(torch.tensor(0.5)) - nn.functional.logsigmoid(torch.tensor(-0.5))


class TestBalancedCrossEntropy(BaseTester):
    def test_smoke(self, device, dtype):
        nc = 4
        input = torch.rand((32, nc, 100), device=device, dtype=dtype)
        valid = torch.randint(0, 2, (32, nc, 100), device=device, dtype=torch.bool)
        target = torch.randint(0, 2, (32, nc, 100), device=device, dtype=dtype)
        bace = BalancedCrossEntropy()
        assert isinstance(bace(input, target, valid), torch.Tensor)

    def test_gradcheck(self, device):
        nc = 4
        input = torch.rand((32, nc, 100), device=device, dtype=torch.float64)
        valid = torch.randint(0, 2, (32, nc, 100), device=device, dtype=torch.bool)
        target = torch.randint(0, 2, (32, nc, 100), device=device, dtype=torch.float64)
        self.gradcheck(
            balanced_cross_entropy_loss_with_logits,
            (input, target, valid),
            dtypes=[torch.float64, torch.float64, torch.bool],
        )

    @pytest.mark.parametrize(
        ["input", "target", "expected"],
        [
            pytest.param(torch.tensor([[[-1e4], [1e4]]]), torch.tensor([[[0.0], [1.0]]]), torch.tensor(0.0)),
            pytest.param(torch.tensor([[[0.5], [0.5]]]), torch.tensor([[[0.0], [1.0]]]), logsig_05_m05),
            pytest.param(torch.tensor([[[0.5], [0.5]]]), torch.tensor([[[0.0], [0.94]]]), logsig_m05),
            pytest.param(torch.tensor([[[0.5], [0.5]]]), torch.tensor([[[0.10], [1.00]]]), logsig_05),
        ],
    )
    def test_value_check(self, input, target, expected):
        bace = BalancedCrossEntropy()
        self.assert_close(torch.sum(bace(input, target)), expected)

    def test_module(self, device, dtype):
        nc = 4
        input = torch.rand((32, nc, 100), device=device, dtype=dtype)
        valid = torch.randint(0, 2, (32, nc, 100), device=device, dtype=torch.bool)
        target = torch.randint(0, 2, (32, nc, 100), device=device, dtype=dtype)
        op = balanced_cross_entropy_loss_with_logits
        op_module = BalancedCrossEntropy()
        self.assert_close(op(input, target, valid), op_module(input, target, valid))
