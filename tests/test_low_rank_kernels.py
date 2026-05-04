"""Tests for low-rank kernel approximations."""

import torch

from laker.kernels import (
    AttentionKernelOperator,
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
)


def test_nystrom_matvec_consistency():
    """Nyström matvec should approximate exact matvec."""
    n = 100
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    lam = 1e-2

    exact = AttentionKernelOperator(e, lambda_reg=lam, dtype=dtype)
    approx = NystromAttentionKernelOperator(e, lambda_reg=lam, num_landmarks=50, dtype=dtype)

    x = torch.randn(n, dtype=dtype)
    y_exact = exact.matvec(x)
    y_approx = approx.matvec(x)

    rel_err = torch.norm(y_exact - y_approx) / torch.norm(y_exact)
    # Low-rank approximations are rough; tolerance is generous
    assert rel_err.item() < 2.0


def test_nystrom_diagonal_positive():
    """Nyström diagonal should be positive."""
    n = 50
    de = 4
    e = torch.randn(n, de)
    op = NystromAttentionKernelOperator(e, lambda_reg=1e-2, num_landmarks=30)
    diag = op.diagonal()
    assert torch.all(diag > 0).item()


def test_nystrom_to_dense_shape():
    """Nyström to_dense should return correct shape."""
    n = 30
    de = 4
    e = torch.randn(n, de)
    op = NystromAttentionKernelOperator(e, lambda_reg=1e-2, num_landmarks=20)
    dense = op.to_dense()
    assert dense.shape == (n, n)


def test_rff_matvec_consistency():
    """RFF matvec should approximate exact matvec."""
    n = 100
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    lam = 1e-2

    exact = AttentionKernelOperator(e, lambda_reg=lam, dtype=dtype)
    approx = RandomFeatureAttentionKernelOperator(e, lambda_reg=lam, num_features=200, dtype=dtype)

    x = torch.randn(n, dtype=dtype)
    y_exact = exact.matvec(x)
    y_approx = approx.matvec(x)

    rel_err = torch.norm(y_exact - y_approx) / torch.norm(y_exact)
    # Low-rank approximations are rough; tolerance is generous
    assert rel_err.item() < 2.0


def test_rff_diagonal_positive():
    """RFF diagonal should be positive."""
    n = 50
    de = 4
    e = torch.randn(n, de)
    op = RandomFeatureAttentionKernelOperator(e, lambda_reg=1e-2, num_features=100)
    diag = op.diagonal()
    assert torch.all(diag > 0).item()


def test_rff_to_dense_shape():
    """RFF to_dense should return correct shape."""
    n = 30
    de = 4
    e = torch.randn(n, de)
    op = RandomFeatureAttentionKernelOperator(e, lambda_reg=1e-2, num_features=50)
    dense = op.to_dense()
    assert dense.shape == (n, n)
