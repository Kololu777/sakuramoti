import torch
from sakuramoti.geometry.grid import generate_grid, normalize_coords


def test_shape_generate_grid():
    grid = generate_grid(5, 4, torch.device("cpu"), homogeneous=False)
    assert grid.shape == (5, 4, 2)


def test_homogeneous_generate_grid():
    grid = generate_grid(5, 4, torch.device("cpu"), homogeneous=True)
    assert grid.shape == (5, 4, 3)


def test_check_value_generate_grid():
    grid = generate_grid(5, 2, torch.device("cpu"), homogeneous=True)
    grid_test = torch.tensor(
        [
            [[0.0000, 0.0000, 1.0000], [2.0000, 0.0000, 1.0000]],
            [[0.0000, 1.2500, 1.0000], [2.0000, 1.2500, 1.0000]],
            [[0.0000, 2.5000, 1.0000], [2.0000, 2.5000, 1.0000]],
            [[0.0000, 3.7500, 1.0000], [2.0000, 3.7500, 1.0000]],
            [[0.0000, 5.0000, 1.0000], [2.0000, 5.0000, 1.0000]],
        ]
    )
    assert torch.allclose(grid, grid_test)


def test_normalize_coords():
    coords = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    normalized_coords = normalize_coords(coords, 2, 2, no_shift=True)
    assert torch.allclose(normalized_coords, torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
