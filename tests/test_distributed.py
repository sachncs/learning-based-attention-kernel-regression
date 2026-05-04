"""Tests for laker.distributed_kernels module."""

import pytest
import torch

from laker.distributed_kernels import DistributedAttentionKernelOperator


def test_distributed_single_device_fallback():
    """DistributedAttentionKernelOperator should fall back when only one device."""
    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 8, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)

    dist_op = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    assert dist_op.single_device

    y_dist = dist_op.matvec(x)
    y_diag = dist_op.diagonal()
    assert y_dist.shape == (n,)
    assert y_diag.shape == (n,)


def test_distributed_matvec_matches_dense():
    """Distributed matvec should match single-device AttentionKernelOperator."""
    from laker.kernels import AttentionKernelOperator

    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 8, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)

    dist_op = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    single_op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)

    y_dist = dist_op.matvec(x)
    y_single = single_op.matvec(x)
    assert torch.allclose(y_dist, y_single, atol=1e-6)


def test_distributed_diagonal_matches_dense():
    """Distributed diagonal should match single-device AttentionKernelOperator."""
    from laker.kernels import AttentionKernelOperator

    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 8, dtype=torch.float64)

    dist_op = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    single_op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)

    d_dist = dist_op.diagonal()
    d_single = single_op.diagonal()
    assert torch.allclose(d_dist, d_single, atol=1e-6)


def test_distributed_to_dense_matches_dense():
    """Distributed to_dense should match single-device AttentionKernelOperator."""
    from laker.kernels import AttentionKernelOperator

    torch.manual_seed(42)
    n = 30
    e = torch.randn(n, 8, dtype=torch.float64)

    dist_op = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    single_op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)

    m_dist = dist_op.to_dense()
    m_single = single_op.to_dense()
    assert torch.allclose(m_dist, m_single, atol=1e-6)


def test_distributed_kernel_eval():
    """Distributed kernel_eval should match single-device AttentionKernelOperator."""
    from laker.kernels import AttentionKernelOperator

    torch.manual_seed(42)
    n = 50
    e = torch.randn(n, 8, dtype=torch.float64)
    x = torch.randn(10, 8, dtype=torch.float64)

    dist_op = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    single_op = AttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)

    k_dist = dist_op.kernel_eval(x)
    k_single = single_op.kernel_eval(x)
    assert torch.allclose(k_dist, k_single, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_distributed_multi_device_matvec():
    """Distributed matvec on CUDA should run without error."""
    torch.manual_seed(42)
    n = 100
    e = torch.randn(n, 8, dtype=torch.float64, device="cuda")
    x = torch.randn(n, dtype=torch.float64, device="cuda")

    dist_op = DistributedAttentionKernelOperator(
        e, lambda_reg=1e-2, master_device=torch.device("cuda")
    )
    y = dist_op.matvec(x)
    assert y.shape == (n,)
    assert y.device.type == "cuda"
