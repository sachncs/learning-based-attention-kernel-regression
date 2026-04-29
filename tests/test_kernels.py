"""Tests for attention kernel operators."""

import torch

from laker.kernels import AttentionKernelOperator


def test_kernel_shape():
    n = 20
    de = 5
    e = torch.randn(n, de)
    op = AttentionKernelOperator(e, lambda_reg=0.1)
    assert op.shape == (n, n)
    assert op.n == n
    assert op.embedding_dim == de


def test_kernel_matvec_consistency():
    """Matrix-free matvec must match explicit dense multiplication."""
    n = 30
    de = 4
    e = torch.randn(n, de)
    lam = 0.05
    op = AttentionKernelOperator(e, lambda_reg=lam)

    x = torch.randn(n)
    y_op = op.matvec(x)

    # Explicit construction
    g = torch.exp(e @ e.T)
    g.diagonal().add_(lam)
    y_dense = g @ x

    torch.testing.assert_close(y_op, y_dense, rtol=1e-5, atol=1e-6)


def test_kernel_matvec_chunked_consistency():
    """Chunked matvec must match full matvec."""
    n = 100
    de = 6
    e = torch.randn(n, de)
    op_full = AttentionKernelOperator(e, lambda_reg=0.1, chunk_size=None)
    op_chunk = AttentionKernelOperator(e, lambda_reg=0.1, chunk_size=16)

    x = torch.randn(n)
    y_full = op_full.matvec(x)
    y_chunk = op_chunk.matvec(x)

    torch.testing.assert_close(y_full, y_chunk, rtol=1e-4, atol=1e-5)


def test_kernel_diagonal():
    n = 15
    e = torch.randn(n, 3)
    lam = 0.2
    op = AttentionKernelOperator(e, lambda_reg=lam)
    diag = op.diagonal()
    sq = torch.sum(e**2, dim=1)
    expected = lam + torch.exp(sq)
    torch.testing.assert_close(diag, expected, rtol=1e-6, atol=1e-8)


def test_kernel_eval():
    n = 10
    m = 5
    de = 4
    e_train = torch.randn(n, de)
    e_query = torch.randn(m, de)
    op = AttentionKernelOperator(e_train, lambda_reg=0.1)
    k = op.kernel_eval(e_query)
    expected = torch.exp(e_query @ e_train.T)
    torch.testing.assert_close(k, expected, rtol=1e-5, atol=1e-6)
