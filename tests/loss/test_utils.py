from torch import Tensor
from sakuramoti.losses.utils import maksed_mean
import torch
from base import BaseTester
import pytest

class TestMaskedMean(BaseTester):
    @pytest.mark.parametrize(["mean_mode", "target_value"], 
                             [pytest.param("reduce_masked_mean", torch.tensor(0.75)), 
                              pytest.param("mean", torch.tensor(0.6))])
    def test_same_as_expect_to_value(self, mean_mode, target_value):
        loss = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        valid = torch.tensor([True, True, True, False, True], dtype=torch.bool)
        value = maksed_mean(loss, valid, mean_mode=mean_mode)
        self.assert_close(value, target_value)
    
    @pytest.mark.parametrize(["mean_mode", "target_value"], 
                             [pytest.param("reduce_masked_mean", torch.tensor(0.8)), 
                              pytest.param("mean", torch.tensor(0.8))])
    def test_check_valid_none_case(self, mean_mode, target_value):
        loss = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        value = maksed_mean(loss, mean_mode=mean_mode)
        self.assert_close(value, target_value)

    def test_exception(self):
        with pytest.raises(ValueError) as errinfo:
            loss = torch.tensor([1.0])
            maksed_mean(loss, mean_mode="not_mean_mode")
        mean_mode_error_message = {"mean", "reduce_masked_mean"}
        assert f"The `mean_mode` must be one of {mean_mode_error_message}, but got not_mean_mode" in str(errinfo)
    