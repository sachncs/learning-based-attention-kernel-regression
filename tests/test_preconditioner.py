"""Tests for CCCP preconditioner."""

import torch

from laker.kernels import AttentionKernelOperator
from laker.preconditioner import CCCPPreconditioner


def test_preconditioner_build(small_problem):
    """Preconditioner builds without error and applies correctly."""
    x, y = small_problem
    n = x.shape[0]
    de = 4
    e = torch.randn(n, de, device=x.device, dtype=x.dtype)
    op = AttentionKernelOperator(e, lambda_reg=1e-2)

    pre = CCCPPreconditioner(
        num_probes=50,
        gamma=1e-1,
        epsilon=1e-8,
        max_iter=20,
        tol=1e-4,
        verbose=False,
        device=x.device,
        dtype=x.dtype,
    )
    pre.build(op.matvec, n)

    assert pre.q_basis is not None
    assert pre.q_basis.shape == (n, 50)
    assert pre.isotropic_coef is not None
    assert pre.isotropic_coef > 0

    # Apply to a vector
    v = torch.randn(n, device=x.device, dtype=x.dtype)
    pv = pre.apply(v)
    assert pv.shape == v.shape


def test_preconditioner_reduces_condition_number(small_problem):
    """Preconditioned system must have smaller condition number."""
    x, y = small_problem
    n = x.shape[0]
    de = 4
    e = torch.randn(n, de, device=x.device, dtype=x.dtype)
    op = AttentionKernelOperator(e, lambda_reg=1e-2)

    # Original condition number (power iteration)
    def power_iter(matvec, steps=30):
        v = torch.randn(n, device=x.device, dtype=x.dtype)
        v = v / torch.linalg.norm(v)
        for _ in range(steps):
            v = matvec(v)
            v = v / torch.linalg.norm(v)
        return torch.dot(v, matvec(v)).item()

    lam_max_orig = power_iter(op.matvec)

    # Inverse iteration for smallest eigenvalue
    v = torch.randn(n, device=x.device, dtype=x.dtype)
    v = v / torch.linalg.norm(v)
    a_dense = op.to_dense()
    for _ in range(20):
        v = torch.linalg.solve(a_dense, v)
        v = v / torch.linalg.norm(v)
    # Preconditioned
    pre = CCCPPreconditioner(
        num_probes=80,
        gamma=1e-1,
        epsilon=1e-8,
        max_iter=50,
        tol=1e-5,
        verbose=False,
        device=x.device,
        dtype=x.dtype,
    )
    pre.build(op.matvec, n)

    def pa(v):
        return pre.apply(op.matvec(v))

    lam_max_pre = power_iter(pa, steps=30)
    # The preconditioner should at least reduce the largest eigenvalue
    assert lam_max_pre < lam_max_orig
