"""Tests for :mod:`laker.solve`."""

import pytest
import torch

from laker.solve import PCG, Descent, Jacobi, Report


def _sym_pd(n, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(n, n, dtype=torch.float64)
    return a @ a.T + torch.eye(n, dtype=torch.float64)


class TestPCG:
    def test_solves_identity_system(self):
        n = 5
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        pcg = PCG(tol=1e-12, max_iter=200, verbose=False)
        x, status = pcg.solve(lambda v: A @ v, lambda v: v, rhs)
        torch.testing.assert_close(A @ x, rhs, atol=1e-6, rtol=1e-6)
        assert status.converged

    def test_zero_rhs_returns_zero(self):
        pcg = PCG(tol=1e-10, verbose=False)
        n = 5
        rhs = torch.zeros(n, dtype=torch.float64)
        x, status = pcg.solve(lambda v: v, lambda v: v, rhs)
        assert status.reason == "zero_rhs"
        assert status.converged
        torch.testing.assert_close(x, rhs)

    def test_breakdown_raises(self):
        # Indefinite operator: p^T A p can go negative.
        n = 4
        a = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        rhs = torch.randn(n, dtype=torch.float64)
        pcg = PCG(tol=1e-12, max_iter=200, verbose=False)
        with pytest.raises(RuntimeError, match="breakdown"):
            pcg.solve(lambda v: a @ v, lambda v: v, rhs)

    def test_rejects_wrong_dim(self):
        pcg = PCG(verbose=False)
        rhs = torch.randn(2, 2, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            pcg.solve(lambda v: v, lambda v: v, rhs)

    def test_max_iter_termination(self):
        n = 50
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        pcg = PCG(tol=1e-20, max_iter=2, verbose=False)
        _, status = pcg.solve(lambda v: A @ v, lambda v: v, rhs)
        assert not status.converged
        assert status.reason == "max_iter"

    def test_2d_batch_rhs(self):
        n, k = 6, 3
        A = _sym_pd(n)
        rhs = torch.randn(n, k, dtype=torch.float64)
        pcg = PCG(tol=1e-12, max_iter=200, verbose=False)
        x, status = pcg.solve(lambda v: A @ v, lambda v: v, rhs)
        torch.testing.assert_close(A @ x, rhs, atol=1e-6, rtol=1e-6)
        assert status.per is not None
        assert len(status.per) == k

    def test_warm_start_matches_cold(self):
        n = 8
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        pcg1 = PCG(tol=1e-12, max_iter=500, verbose=False)
        x1, _ = pcg1.solve(lambda v: A @ v, lambda v: v, rhs)
        x0 = torch.randn(n, dtype=torch.float64)
        pcg2 = PCG(tol=1e-12, max_iter=500, verbose=False)
        x2, _ = pcg2.solve(lambda v: A @ v, lambda v: v, rhs, x0=x0)
        torch.testing.assert_close(x1, x2, atol=1e-6, rtol=1e-6)

    def test_iterations_counter(self):
        n = 6
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        pcg = PCG(tol=1e-12, max_iter=500, verbose=False)
        pcg.solve(lambda v: A @ v, lambda v: v, rhs)
        assert pcg.iterations > 0

    def test_float32_well_conditioned_converges(self):
        torch.manual_seed(0)
        n = 1000
        A = torch.rand(n, n, dtype=torch.float32)
        A = A @ A.T + torch.eye(n, dtype=torch.float32)
        rhs = torch.randn(n, dtype=torch.float32)
        pcg = PCG(tol=1e-5, max_iter=500, verbose=False)
        _, status = pcg.solve(lambda v: A @ v, lambda v: v, rhs)
        assert status.converged, (
            f"float32 PCG failed: reason={status.reason}, res={status.residual}"
        )


class TestDescent:
    def test_solves_simple_system(self):
        n = 5
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        gd = Descent(tol=1e-6, max_iter=5000, verbose=False)
        x = gd.solve(lambda v: A @ v, rhs)
        residual = torch.linalg.norm(A @ x - rhs).item() / torch.linalg.norm(rhs).item()
        assert residual < 1e-3

    def test_warm_start(self):
        n = 5
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        gd = Descent(tol=1e-6, max_iter=5000, verbose=False)
        x0 = torch.randn(n, dtype=torch.float64)
        x = gd.solve(lambda v: A @ v, rhs, x0=x0)
        assert x.shape == (n,)


class TestJacobi:
    def test_apply(self):
        d = torch.tensor([2.0, 4.0, 1.0], dtype=torch.float64)
        j = Jacobi(d)
        x = torch.tensor([4.0, 8.0, 3.0], dtype=torch.float64)
        out = j.apply(x)
        torch.testing.assert_close(out, torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64))

    def test_apply_handles_zero(self):
        d = torch.tensor([0.0, 1.0], dtype=torch.float64)
        j = Jacobi(d)
        out = j.apply(torch.tensor([1.0, 1.0], dtype=torch.float64))
        assert torch.isfinite(out).all()

    def test_2d_input(self):
        d = torch.tensor([2.0, 1.0], dtype=torch.float64)
        j = Jacobi(d)
        x = torch.tensor([[4.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
        out = j.apply(x)
        assert out.shape == (2, 2)


class TestReport:
    def test_construction(self):
        r = Report(converged=True, iterations=10, residual=1e-8, reason="converged")
        assert r.converged
        assert r.iterations == 10
        assert r.reason == "converged"
        assert r.per is None
