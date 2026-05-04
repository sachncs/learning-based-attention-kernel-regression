"""Tests for synthetic data generation utilities."""

import torch

from laker.data import generate_grid, generate_radio_field


def test_generate_radio_field_shape():
    n = 50
    locs = torch.rand(n, 2) * 100.0
    tx = torch.tensor([[30.0, 70.0]])
    pwr = torch.tensor([-40.0])
    clean, noisy = generate_radio_field(locs, tx, pwr)
    assert clean.shape == (n,)
    assert noisy.shape == (n,)
    assert noisy.dtype == locs.dtype
    assert noisy.device == locs.device


def test_generate_radio_field_seed_reproducibility():
    n = 20
    locs = torch.rand(n, 2) * 100.0
    tx = torch.tensor([[50.0, 50.0]])
    pwr = torch.tensor([-40.0])
    _, noisy1 = generate_radio_field(locs, tx, pwr, seed=42)
    _, noisy2 = generate_radio_field(locs, tx, pwr, seed=42)
    torch.testing.assert_close(noisy1, noisy2)


def test_generate_grid():
    grid = generate_grid((0.0, 100.0, 0.0, 100.0), grid_size=10)
    assert grid.shape == (100, 2)
    assert grid[0, 0].item() == 0.0
    assert grid[-1, 0].item() == 100.0


def test_generate_radio_field_bad_locations():
    """1-D locations should raise ValueError."""
    import pytest

    locs = torch.randn(5)
    tx = torch.tensor([[1.0, 1.0]])
    pwr = torch.tensor([-40.0])
    with pytest.raises(ValueError, match="locations must be 2-D"):
        generate_radio_field(locs, tx, pwr)


def test_generate_radio_field_bad_transmitters():
    """1-D transmitters should raise ValueError."""
    import pytest

    locs = torch.rand(5, 2)
    tx = torch.tensor([1.0, 1.0])
    pwr = torch.tensor([-40.0])
    with pytest.raises(ValueError, match="transmitters must be 2-D"):
        generate_radio_field(locs, tx, pwr)


def test_generate_radio_field_bad_powers():
    """2-D powers should raise ValueError."""
    import pytest

    locs = torch.rand(5, 2)
    tx = torch.tensor([[1.0, 1.0]])
    pwr = torch.tensor([[-40.0]])
    with pytest.raises(ValueError, match="powers must be 1-D"):
        generate_radio_field(locs, tx, pwr)


def test_generate_radio_field_mismatched_tx_powers():
    """Mismatched tx and powers lengths should raise ValueError."""
    import pytest

    locs = torch.rand(5, 2)
    tx = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    pwr = torch.tensor([-40.0])
    with pytest.raises(ValueError, match="transmitters and powers must have same length"):
        generate_radio_field(locs, tx, pwr)


def test_generate_radio_field_mismatched_dimensions():
    """Mismatched spatial dimensions should raise ValueError."""
    import pytest

    locs = torch.rand(5, 2)
    tx = torch.tensor([[1.0, 1.0, 1.0]])
    pwr = torch.tensor([-40.0])
    with pytest.raises(
        ValueError,
        match="locations and transmitters must have same spatial dimension",
    ):
        generate_radio_field(locs, tx, pwr)
