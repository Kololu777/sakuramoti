import torch
import pytest
from sakuramoti.geometry.grid import normalize_coords

from tests.base import assert_close


def test_exception():
    coords = torch.rand(10, 10, 1)
    with pytest.raises(AssertionError) as errinfo:
        normalize_coords(coords, 10, 10)
    assert f"shape of coords must be (*, 2), but got {coords.shape}" in str(errinfo)


@pytest.mark.parametrize(
    "no_shift, gt",
    [
        (
            True,
            torch.tensor(
                [
                    [[0.0000, 0.0000], [2.0000, 0.0000]],
                    [[0.0000, 1.0000], [2.0000, 1.0000]],
                    [[0.0000, 2.0000], [2.0000, 2.0000]],
                ]
            ),
        ),
        (
            False,
            torch.tensor(
                [
                    [[-1.0000, -1.0000], [1.0000, -1.0000]],
                    [[-1.0000, 0.0000], [1.0000, 0.0000]],
                    [[-1.0000, 1.0000], [1.0000, 1.0000]],
                ]
            ),
        ),
    ],
)
def test_value_check(no_shift, gt):
    coords = torch.tensor(
        [
            [[0.0000, 0.0000], [2.0, 0.0000]],
            [[0.0000, 1.5000], [2.0000, 1.5000]],
            [[0.0000, 3.0000], [2.0000, 3.0000]],
        ]
    )
    normalized_coords = normalize_coords(coords, 3, 2, no_shift=no_shift)
    assert_close(normalized_coords, gt)
