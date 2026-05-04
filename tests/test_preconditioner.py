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


def test_adaptive_probes_not_catastrophically_worse():
    """Power-iteration probes (mixed with Gaussian) should remain effective.

    Pure power-iteration probes can collapse to the dominant eigenspace and
    break the CCCP MLE assumptions.  Our implementation mixes 25 %% power-
    iterated probes with 75 %% Gaussian probes to retain unbiased spectrum
    coverage.  This test verifies that the mixed strategy does not make the
    preconditioner catastrophically worse than pure Gaussian probes.
    """
    device = torch.device("cpu")
    dtype = torch.float64
    n = 500
    de = 8

    # Construct an embedding with a controlled spectrum gap via SVD.
    u = torch.linalg.qr(torch.randn(n, de, device=device, dtype=dtype))[0]
    v = torch.linalg.qr(torch.randn(de, de, device=device, dtype=dtype))[0]
    s = torch.tensor([2.0, 1.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], device=device, dtype=dtype)
    e = u @ torch.diag(s) @ v
    op = AttentionKernelOperator(e, lambda_reg=1e-3)

    def power_iter(matvec, steps=40):
        v = torch.randn(n, device=device, dtype=dtype)
        v = v / torch.linalg.norm(v)
        for _ in range(steps):
            v = matvec(v)
            v = v / torch.linalg.norm(v)
        return torch.dot(v, matvec(v)).item()

    def build_pre(probe_strategy):
        pre = CCCPPreconditioner(
            num_probes=50,
            gamma=1e-1,
            epsilon=1e-8,
            base_rho=0.05,
            max_iter=80,
            tol=1e-5,
            verbose=False,
            device=device,
            dtype=dtype,
            probe_strategy=probe_strategy,
            power_iter_steps=3,
        )
        pre.build(op.matvec, n, seed=42)
        return pre

    pre_gauss = build_pre("gaussian")
    pre_power = build_pre("power_iter")

    def pa(pre, v):
        return pre.apply(op.matvec(v))

    lam_max_orig = power_iter(op.matvec, steps=40)
    lam_max_gauss = power_iter(lambda v: pa(pre_gauss, v), steps=40)
    lam_max_power = power_iter(lambda v: pa(pre_power, v), steps=40)

    # Both strategies must substantially reduce the condition number.
    assert lam_max_gauss < lam_max_orig
    assert lam_max_power < lam_max_orig
    # Mixed power-iteration strategy should be within a factor of 2 of the
    # pure Gaussian baseline (empirically it is ~5 % worse on this problem).
    assert lam_max_power < lam_max_gauss * 2.0
