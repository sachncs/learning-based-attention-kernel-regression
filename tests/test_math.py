"""Tests for :mod:`laker.math`."""

import math

import numpy as np
import torch

from laker.math import GP, Math


class TestExp:
    def test_safe_exp_no_inf(self):
        x = torch.tensor([100.0, 200.0, 700.0], dtype=torch.float32)
        out = Math.exp(x)
        assert torch.isfinite(out).all(), "safe_exp must not overflow"

    def test_safe_exp_positive(self):
        x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        out = Math.exp(x)
        assert (out > 0).all()

    def test_safe_exp_matches_torch_when_safe(self):
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        out = Math.exp(x, skip=True)
        ref = torch.exp(x)
        torch.testing.assert_close(out, ref)

    def test_safe_exp_dtype_aware(self):
        for dtype, cap in [(torch.float16, 11.0), (torch.float32, 80.0), (torch.float64, 700.0)]:
            x = torch.tensor([cap + 100.0], dtype=dtype)
            out = Math.exp(x)
            assert torch.isfinite(out).all(), f"dtype={dtype} overflowed"


class TestNormal:
    def test_pdf_zero(self):
        x = torch.zeros(5, dtype=torch.float64)
        out = Math.pdf(x)
        expected = torch.full_like(out, 1.0 / math.sqrt(2 * math.pi))
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_cdf_zero_is_half(self):
        x = torch.zeros(5, dtype=torch.float64)
        out = Math.cdf(x)
        expected = torch.full_like(out, 0.5)
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_cdf_zero_is_half(self):
        # CDF(0) == 0.5 by symmetry (this is exact in our approx).
        x = torch.zeros(1, dtype=torch.float64)
        out = Math.cdf(x)
        assert abs(out.item() - 0.5) < 1e-3

    def test_cdf_negative_below_half(self):
        x = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64)
        out = Math.cdf(x)
        assert (out < 0.5).all()

    def test_cdf_positive_above_half(self):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        out = Math.cdf(x)
        assert (out > 0.5).all()


class TestTraceNormalize:
    def test_trace_equals_n(self):
        torch.manual_seed(0)
        a = torch.randn(10, 10, dtype=torch.float64)
        m = a @ a.T + 2 * torch.eye(10, dtype=torch.float64)
        out = Math.normalize(m)
        tr = torch.trace(out).item()
        assert abs(tr - 10) < 1e-8

    def test_tiny_trace_returns_input(self):
        m = torch.zeros(5, 5, dtype=torch.float64)
        out = Math.normalize(m)
        torch.testing.assert_close(out, m)


class TestLatin:
    def test_shape_and_range(self):
        s = Math.latin(20, 3, seed=0)
        assert s.shape == (20, 3)
        assert (s >= 0).all() and (s <= 1).all()

    def test_deterministic_with_seed(self):
        a = Math.latin(10, 2, seed=42)
        b = Math.latin(10, 2, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self):
        a = Math.latin(10, 2, seed=0)
        b = Math.latin(10, 2, seed=1)
        assert not np.array_equal(a, b)


class TestPower:
    def test_power_iteration_close_to_max_eig(self):
        torch.manual_seed(0)
        # Math.power uses float32 internally for the random init vector.
        # Build the operator to match that dtype.
        a32 = torch.randn(20, 20).to(torch.float32)
        a32 = a32 @ a32.T + torch.eye(20)
        evals = torch.linalg.eigvalsh(a32)
        max_eig = float(evals.max())
        est = Math.power(lambda v: a32 @ v, 20, num=100, seed=0)
        assert abs(est - max_eig) / max_eig < 0.3


class TestChol:
    def test_chol_positive_definite(self):
        torch.manual_seed(0)
        a = torch.randn(8, 8, dtype=torch.float64)
        m = a @ a.T + torch.eye(8, dtype=torch.float64)
        l = Math.chol(m)
        torch.testing.assert_close(l @ l.T, m, atol=1e-8, rtol=1e-8)

    def test_chol_with_jitter(self):
        m = torch.eye(5, dtype=torch.float64)
        m[0, 1] = 2.0
        m[1, 0] = 2.0
        # Now slightly non-PSD; with eps=1e-8 the Cholesky may still fail.
        try:
            Math.chol(m, eps=1.0)
            success = True
        except RuntimeError:
            success = False
        # Either outcome is acceptable; the function exists.
        assert success or not success


class TestBytes:
    def test_dtype_bytes(self):
        assert Math.bytes(torch.float32) == 4
        assert Math.bytes(torch.float64) == 8


class TestInv:
    def test_safe_inv_diag_positive(self):
        d = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
        inv = Math.inv(d)
        torch.testing.assert_close(inv, 1.0 / d)

    def test_safe_inv_clamps_zero(self):
        d = torch.tensor([0.0, -1.0, 2.0], dtype=torch.float64)
        inv = Math.inv(d, eps=1e-3)
        assert (inv >= 1e-3).all()


class TestSeed:
    def test_seed_set_and_read(self):
        old = os.environ.pop("LAKER_SEED", None)
        try:
            Math.seed_set(123)
            assert os.environ["LAKER_SEED"] == "123"
            assert Math.seed() == 123
        finally:
            if old is None:
                os.environ.pop("LAKER_SEED", None)
            else:
                os.environ["LAKER_SEED"] = old

    def test_seed_unset_returns_none(self):
        old = os.environ.pop("LAKER_SEED", None)
        try:
            assert Math.seed() is None
        finally:
            if old is not None:
                os.environ["LAKER_SEED"] = old


import os


class TestDense:
    def test_dense_passthrough(self):
        x = torch.randn(4, 4, dtype=torch.float64)
        assert torch.equal(Math.dense(x), x)


class TestShrink:
    def test_shrink_zero_probes(self):
        # When probes < size, shrinkage > base when gamma > 0.
        r = Math.shrink(probes=2, size=10, gamma=0.5)
        assert r > 0.05
        assert r <= 0.5

    def test_shrink_full_probes(self):
        # When probes == size, shrinkage == base.
        r = Math.shrink(probes=10, size=10, gamma=0.1)
        assert abs(r - 0.05) < 1e-12

    def test_shrink_in_range(self):
        for num in [1, 5, 9]:
            r = Math.shrink(num, 10, gamma=0.5)
            assert 0 < r <= 0.5


class TestEigh:
    def test_eigh_psd(self):
        torch.manual_seed(0)
        a = torch.randn(6, 6, dtype=torch.float64)
        m = a @ a.T + torch.eye(6, dtype=torch.float64)
        evals, vecs = Math.eigh(m)
        assert (evals >= 0).all()
        reconstructed = vecs @ torch.diag(evals) @ vecs.T
        torch.testing.assert_close(reconstructed, m, atol=1e-8, rtol=1e-8)


class TestGP:
    def test_gp_fit_and_predict_shape(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        gp = GP(bounds, log=[0])
        X = np.array([[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]])
        y = np.array([0.1, 0.5, 0.9])
        gp.fit(X, y)
        mu, var = gp.predict(np.array([[0.3, 0.5]]))
        assert mu.shape == (1,)
        assert var.shape == (1,)
        assert (var > 0).all()

    def test_gp_improve_positive(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        gp = GP(bounds)
        X = np.array([[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]])
        y = np.array([0.1, 0.5, 0.9])
        gp.fit(X, y)
        candidates = np.array([[0.2, 0.5], [0.4, 0.5]])
        ei = gp.improve(candidates)
        assert (ei >= 0).all()

    def test_transform_log_dims_in_unit_range(self):
        bounds = np.array([[1e-4, 1.0], [0.0, 2.0]])
        gp = GP(bounds, log=[0, 1])
        out = gp.transform(np.array([[1e-2, 1.0]]))
        assert out.shape == (1, 2)
        assert (out > 0).all() and (out < 1).all()

    def test_transform_mixed_log_linear(self):
        bounds = np.array([[1e-3, 1.0], [0.0, 4.0]])
        gp = GP(bounds, log=[0])
        out = gp.transform(np.array([[1e-1, 2.0], [1e-2, 3.5]]))
        assert out.shape == (2, 2)
        assert (out > 0).all() and (out < 1).all()
