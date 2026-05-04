"""Tests for iterative solvers."""

import torch

from laker.kernels import AttentionKernelOperator
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import (
    GradientDescent,
    JacobiPreconditioner,
    PreconditionedConjugateGradient,
)


def test_pcg_solves_exactly():
    """PCG with exact preconditioner should converge in one iteration."""
    n = 20
    a_dense = torch.diag(torch.arange(1, n + 1, dtype=torch.float64))
    b = torch.randn(n, dtype=torch.float64)
    x_exact = torch.linalg.solve(a_dense, b)

    def op(v):
        return a_dense @ v

    def pre(v):
        return torch.linalg.solve(a_dense, v)

    pcg = PreconditionedConjugateGradient(tol=1e-12, max_iter=n, verbose=False)
    x = pcg.solve(op, pre, b)
    torch.testing.assert_close(x, x_exact, rtol=1e-5, atol=1e-6)


def test_pcg_with_learned_preconditioner():
    """PCG with LAKER preconditioner solves attention kernel system."""
    n = 50
    de = 4
    dtype = torch.float64
    e = torch.randn(n, de, dtype=dtype)
    lam = 1e-2
    op = AttentionKernelOperator(e, lambda_reg=lam, dtype=dtype)
    b = torch.randn(n, dtype=dtype)

    pre = CCCPPreconditioner(
        num_probes=50,
        gamma=1e-1,
        max_iter=30,
        tol=1e-5,
        verbose=False,
        dtype=dtype,
    )
    pre.build(op.matvec, n)

    pcg = PreconditionedConjugateGradient(tol=1e-8, max_iter=200, verbose=False)
    x = pcg.solve(op.matvec, pre.apply, b)

    res = torch.linalg.norm(op.matvec(x) - b) / torch.linalg.norm(b)
    assert res.item() < 5e-2


def test_jacobi_pcg():
    """Jacobi PCG should also converge for diagonally dominant systems."""
    n = 30
    de = 4
    e = torch.randn(n, de)
    lam = 1.0  # strong diagonal dominance
    op = AttentionKernelOperator(e, lambda_reg=lam)
    b = torch.randn(n)

    jac = JacobiPreconditioner(op.diagonal())
    pcg = PreconditionedConjugateGradient(tol=1e-8, max_iter=200, verbose=False)
    x = pcg.solve(op.matvec, jac.apply, b)

    res = torch.linalg.norm(op.matvec(x) - b) / torch.linalg.norm(b)
    assert res.item() < 1e-5


def test_gd_convergence():
    """Gradient descent should converge for well-conditioned systems."""
    n = 20
    a_dense = torch.eye(n, dtype=torch.float64) * 2.0 + torch.ones(n, n, dtype=torch.float64) * 0.1
    b = torch.randn(n, dtype=torch.float64)

    def op(v):
        return a_dense @ v

    gd = GradientDescent(step_size=0.4, tol=1e-4, max_iter=5000, verbose=False)
    x = gd.solve(op, b)

    res = torch.linalg.norm(op(x) - b) / torch.linalg.norm(b)
    assert res.item() < 1e-3
