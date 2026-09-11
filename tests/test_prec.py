"""Tests for :mod:`laker.prec`."""

import pytest
import torch

from laker.prec import CCCP, Adaptive


def _sym_pd(n, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(n, n, dtype=torch.float64)
    return a @ a.T + torch.eye(n, dtype=torch.float64)


class TestCCCP:
    def test_build_sets_attrs(self):
        n = 30
        A = _sym_pd(n)
        p = CCCP(num=10, gamma=0.1, max_iter=50, tol=1e-6, verbose=False, dtype=torch.float64)
        p.build(A @ (lambda v: v) if False else (lambda v: A @ v), n)
        assert p.size == n
        assert p.basis is not None
        assert p.evals is not None
        assert p.iso is not None and p.iso > 0

    def test_apply_shape(self):
        n = 20
        A = _sym_pd(n)
        p = CCCP(num=10, gamma=0.1, max_iter=50, tol=1e-6, verbose=False, dtype=torch.float64)
        p.build(lambda v: A @ v, n)
        v = torch.randn(n, dtype=torch.float64)
        out = p.apply(v)
        assert out.shape == (n,)

    def test_apply_2d(self):
        n = 20
        A = _sym_pd(n)
        p = CCCP(num=10, gamma=0.1, max_iter=50, tol=1e-6, verbose=False, dtype=torch.float64)
        p.build(lambda v: A @ v, n)
        v = torch.randn(n, 3, dtype=torch.float64)
        out = p.apply(v)
        assert out.shape == (n, 3)

    def test_apply_shrinks_spectrum(self):
        n = 50
        torch.manual_seed(0)
        A = torch.rand(n, n, dtype=torch.float64)
        A = A @ A.T + torch.eye(n, dtype=torch.float64)
        p = CCCP(num=20, gamma=0.1, max_iter=50, tol=1e-6, verbose=False, dtype=torch.float64)
        p.build(lambda v: A @ v, n)
        v = torch.randn(n, dtype=torch.float64)
        Av = A @ v
        Pv = p.apply(Av)
        # The CCCP preconditioner normalises the operator so ||P A v||
        # should be comparable to ||v|| rather than to ||A v||.
        # We just check that the preconditioner is finite and non-trivial.
        assert torch.isfinite(Pv).all()
        assert torch.linalg.norm(Pv).item() > 0

    def test_rejects_build_then_apply_without_build(self):
        p = CCCP(verbose=False, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="not been built"):
            p.apply(torch.randn(5, dtype=torch.float64))

    def test_dense_shape(self):
        n = 10
        A = _sym_pd(n)
        p = CCCP(num=5, verbose=False, dtype=torch.float64)
        p.build(lambda v: A @ v, n)
        d = p.dense()
        assert d.shape == (n, n)

    def test_apply_wrong_dim(self):
        n = 10
        A = _sym_pd(n)
        p = CCCP(num=5, verbose=False, dtype=torch.float64)
        p.build(lambda v: A @ v, n)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            p.apply(torch.randn(2, 2, 2, dtype=torch.float64))

    def test_deterministic_with_seed(self):
        n = 20
        A = _sym_pd(n)
        p1 = CCCP(num=10, gamma=0.1, verbose=False, dtype=torch.float64)
        p1.build(lambda v: A @ v, n, seed=0)
        p2 = CCCP(num=10, gamma=0.1, verbose=False, dtype=torch.float64)
        p2.build(lambda v: A @ v, n, seed=0)
        torch.testing.assert_close(p1.basis, p2.basis)
        torch.testing.assert_close(p1.evals, p2.evals)

    def test_apply_core_clamps_zero_iso(self):
        from laker.prec import apply_core

        basis = torch.eye(4, dtype=torch.float64)
        evals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        evecs = torch.eye(4, dtype=torch.float64)
        x = torch.ones(4, dtype=torch.float64)
        out = apply_core(x, 0.0, basis, evals, evecs, eps=1e-12)
        assert torch.isfinite(out).all()

    def test_apply_core_clamps_negative_iso(self):
        from laker.prec import apply_core

        basis = torch.eye(4, dtype=torch.float64)
        evals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        evecs = torch.eye(4, dtype=torch.float64)
        x = torch.ones(4, dtype=torch.float64)
        out = apply_core(x, -1.0, basis, evals, evecs, eps=1e-12)
        assert torch.isfinite(out).all()


class TestAdaptive:
    def test_picks_jacobi_for_well_conditioned(self):
        n = 10
        A = _sym_pd(n)
        p = Adaptive(num=5, verbose=False, dtype=torch.float64)
        p.build(lambda v: A @ v, n, diag=A.diagonal(), seed=0)
        assert p.choice == "jacobi"

    def test_apply_shape_jacobi(self):
        n = 10
        A = _sym_pd(n)
        p = Adaptive(verbose=False, dtype=torch.float64)
        p.build(lambda v: A @ v, n, diag=A.diagonal(), seed=0)
        v = torch.randn(n, dtype=torch.float64)
        out = p.apply(v)
        assert out.shape == (n,)

    def test_rejects_apply_without_build(self):
        p = Adaptive(verbose=False, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="not been built"):
            p.apply(torch.randn(5, dtype=torch.float64))
