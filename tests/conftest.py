"""Shared pytest fixtures."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Default compute device for tests."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dtype():
    """Default floating-point dtype for tests."""
    return torch.float64


@pytest.fixture
def small_problem(device, dtype):
    """Return a tiny synthetic problem for unit tests."""
    n = 50
    dx = 2
    x = torch.rand(n, dx, device=device, dtype=dtype) * 100.0
    y = torch.randn(n, device=device, dtype=dtype)
    return x, y
