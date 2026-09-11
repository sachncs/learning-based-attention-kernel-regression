"""Tests for :mod:`laker.distributed`."""

from unittest.mock import patch

import pytest
import torch

from laker.distributed import Distributed


def _sym_pd(n, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(n, n, dtype=torch.float64)
    return a @ a.T + torch.eye(n, dtype=torch.float64)


class TestSingleDevice:
    def test_single_device_path(self):
        n = 15
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        assert op.single
        assert op.devices == [torch.device("cpu")]
        v = torch.randn(n, dtype=torch.float64)
        out = op.matvec(v)
        assert out.shape == (n,)
        assert torch.isfinite(out).all()

    def test_single_device_matvec_matches_inner(self):
        n = 12
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        v = torch.randn(n, dtype=torch.float64)
        torch.testing.assert_close(op.matvec(v), op.local_op.matvec(v))

    def test_diag_matches_inner(self):
        n = 10
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        torch.testing.assert_close(op.diag(), op.local_op.diag())

    def test_dense_matches_inner(self):
        n = 10
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        torch.testing.assert_close(op.dense(), op.local_op.dense())

    def test_eval_matches_inner(self):
        n = 10
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        q = torch.randn(5, 4, dtype=torch.float64)
        torch.testing.assert_close(op.eval(q), op.local_op.eval(q))


class TestSlicing:
    def test_sum_of_shards_equals_full(self):
        n = 12
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        # Force multi-device code path by patching single.
        op.single = False
        # Build artificial per-device shards.
        from laker.kernel import Exact

        op.ops = [
            Exact(embeddings=e[: n // 2], lam=0.1, dtype=torch.float64),
            Exact(embeddings=e[n // 2 :], lam=0.1, dtype=torch.float64),
        ]
        op.shards = [e[: n // 2], e[n // 2 :]]
        op.slices = [(0, n // 2), (n // 2, n)]
        op.sizes = [n // 2, n - n // 2]
        v = torch.randn(n, dtype=torch.float64)
        out = op.matvec(v)
        ref = torch.exp(e @ e.T) @ v + 0.1 * v
        torch.testing.assert_close(out, ref, atol=1e-8, rtol=1e-8)


@pytest.mark.cuda
class TestMultiDeviceMocked:
    """Multi-device path exercised under a CUDA-availability mock.

    These tests don't require an actual GPU; they mock
    ``torch.cuda.is_available`` and ``torch.cuda.device_count`` to drive
    the multi-device code path. Deselected on CPU-only CI by default;
    opt in with ``pytest -m cuda``.
    """

    def test_multi_device_construction_with_mocked_cuda(self):
        n = 8
        e = torch.randn(n, 4, dtype=torch.float64)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch.object(Distributed, "shard", lambda self, x: None),
        ):
            op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        assert op.single is False
        assert len(op.devices) == 2

    def test_multi_device_matvec_matches_reference(self):
        from laker.kernel import Exact

        n = 10
        e = torch.randn(n, 4, dtype=torch.float64)

        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        op.single = False
        op.devices = [torch.device("cpu"), torch.device("cpu")]
        op.ops = [
            Exact(embeddings=e[:5], lam=0.1, dtype=torch.float64, device=torch.device("cpu")),
            Exact(embeddings=e[5:], lam=0.1, dtype=torch.float64, device=torch.device("cpu")),
        ]
        op.shards = [e[:5], e[5:]]
        op.slices = [(0, 5), (5, n)]
        op.sizes = [5, 5]
        v = torch.randn(n, dtype=torch.float64)
        out = op.matvec(v)
        ref = torch.exp(e @ e.T) @ v + 0.1 * v
        torch.testing.assert_close(out, ref, atol=1e-8, rtol=1e-8)


class TestShape:
    def test_shape_attributes(self):
        n = 12
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        assert op.size == n
        assert op.dim == 4
        assert op.shape == (n, n)
        assert op.dtype == torch.float64
