import torch
import pytest
from sakuramoti.metrics.end_point_error import end_point_error
from base import BaseTester
class TestEPEMetrics(BaseTester):
    @pytest.mark.parametrize(["dim", "valid_dim"], [pytest.param((32, 5, 64, 64), (32, 64, 64)), 
                                                pytest.param((32, 5, 64), (32, 64))
                                                ])
    def test_smoke(self, dim, valid_dim, device, dtype):
        pred = torch.rand(dim, device=device, dtype=dtype) #(B, T, H, W) or (B, T, N)
        target = torch.rand(dim, device=device, dtype=dtype) #(B, T, H, W) or (B, T, N)
        valid = torch.randint(0, 2, size=valid_dim, device=device, dtype=torch.bool) #(B, T, H, W) or (B, T, N)
        metrics = end_point_error(pred, target, valid)
        assert isinstance(metrics, dict)
        assert list(metrics.keys()) == ["epe", "1px", "3px", "5px"]


    def test_exception(self, device, dtype):
        pred = torch.rand((32, 2, 64), device=device, dtype=dtype) #(B, T, H, W) or (B, T, N)
        target = torch.rand((32, 2, 64), device=device, dtype=dtype) #(B, T, H, W) or (B, T, N)
        valid = torch.randint(0, 2, size=(32, 1), device=device, dtype=torch.bool) #(B, T, H, W) or (B, T, N)
        with pytest.raises(AssertionError) as errinfo:
            end_point_error(pred, target, valid)
        assert f"Shape of `pred` and `valid` are not paired. Expected to shape of `pred`, `valid` is (B, T, ...), (B, ...) respectivly. but Get to shape of `pred`, `valid` is {torch.Size((32, 2, 64))}, {torch.Size((32, 1))}." in str(errinfo)

    def test_epe_all_equal_one_batch(self, device, dtype):
        pred = torch.tensor([[[[0.1]], [[0.2]], [[0.3]], [[0.4]]]], device=device, dtype=dtype)
        target = torch.tensor([[[[0.1]], [[0.2]], [[0.3]], [[0.4]]]], device=device, dtype=dtype)
        metrics = end_point_error(pred, target)
        self.assert_close(metrics["epe"], 0.00)
        self.assert_close(metrics["1px"], 1.00)
        self.assert_close(metrics["3px"], 1.00)
        self.assert_close(metrics["5px"], 1.00)
    
    def test_epe_all_equal_two_batch(self, device, dtype):
        pred = torch.tensor([[[[0.1]], [[0.2]], [[0.3]], [[0.4]]],
                            [[[0.1]], [[0.2]], [[0.3]], [[0.4]]]
                            ], device=device, dtype=dtype)
        target = torch.tensor([[[[0.1]], [[0.2]], [[0.3]], [[0.4]]],
                            [[[0.1]], [[0.2]], [[0.3]], [[0.4]]]
                            ],device=device, dtype=dtype)
        
        metrics = end_point_error(pred, target)
        self.assert_close(metrics["epe"], 0.00)
        self.assert_close(metrics["1px"], 1.00)
        self.assert_close(metrics["3px"], 1.00)
        self.assert_close(metrics["5px"], 1.00)

    def test_epe_one_miss_one_batch(self, device, dtype):
        pred = torch.tensor([[[[0.1]], [[0.2]], [[0.3]], [[0.4]]]], device=device, dtype=dtype)
        target = torch.tensor([[[[2.1]], [[0.2]], [[0.3]], [[0.4]]]], device=device, dtype=dtype)
        metrics = end_point_error(pred, target)
        self.assert_close(metrics["epe"], 2.00)
        self.assert_close(metrics["1px"], 0.00)
        self.assert_close(metrics["3px"], 1.00)
        self.assert_close(metrics["5px"], 1.00)
        
    
    def test_valid_check_one_batch(self, device, dtype):
        pred = torch.tensor([[[[0.1]], [[0.2]], [[0.3]], [[0.4]]],
                            [[[0.1]], [[0.2]], [[0.3]], [[0.4]]]
                            ], device=device, dtype=dtype)
        target = torch.tensor([[[[0.1]], [[0.2]], [[0.3]], [[0.4]]],
                            [[[0.1]], [[0.2]], [[0.3]], [[0.4]]]
                            ],device=device, dtype=dtype)
        valid = torch.tensor([[[True]], [[False]]], device=device)
        
        metrics = end_point_error(pred, target, valid)
        self.assert_close(metrics["epe"], 0.00)
        self.assert_close(metrics["1px"], 1.00)
        self.assert_close(metrics["3px"], 1.00)
        self.assert_close(metrics["5px"], 1.00)
    
    