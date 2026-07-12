"""Tests for attention kernel operators."""

import torch

from laker.kernels import AttentionKernelOperator


def test_kernel_shape():
    """Test that AttentionKernelOperator reports correct shape attributes."""
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

    torch.testing.assert_close(y_op, y_dense, rtol=1e-4, atol=1e-5)


def test_kernel_matvec_chunked_consistency():
    """Chunked matvec must match full matvec."""
    torch.manual_seed(42)
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
    """Test that diagonal matches the analytical formula."""
    n = 15
    e = torch.randn(n, 3)
    lam = 0.2
    op = AttentionKernelOperator(e, lambda_reg=lam)
    diag = op.diagonal()
    sq = torch.sum(e**2, dim=1)
    expected = lam + torch.exp(sq)
    torch.testing.assert_close(diag, expected, rtol=1e-6, atol=1e-8)


def test_kernel_eval():
    """Test that kernel_eval matches explicit kernel computation."""
    n = 10
    m = 5
    de = 4
    e_train = torch.randn(n, de)
    e_query = torch.randn(m, de)
    op = AttentionKernelOperator(e_train, lambda_reg=0.1)
    k = op.kernel_eval(e_query)
    expected = torch.exp(e_query @ e_train.T)
    torch.testing.assert_close(k, expected, rtol=1e-5, atol=1e-6)


def test_spectral_kernel_matvec_consistency():
    """Spectral kernel matvec must match explicit dense form."""
    torch.manual_seed(42)
    n = 30
    de = 4
    e = torch.randn(n, de)
    lam = 0.05
    from laker.kernels import SpectralAttentionKernelOperator

    op = SpectralAttentionKernelOperator(e, lambda_reg=lam, num_knots=5)
    x = torch.randn(n)
    y_op = op.matvec(x)

    # Explicit: K = U @ diag(spectrum) @ U^T
    k_dense = op.u_matrix @ torch.diag(op.spectrum) @ op.u_matrix.T
    k_dense.diagonal().add_(lam)
    y_dense = k_dense @ x

    torch.testing.assert_close(y_op, y_dense, rtol=1e-4, atol=1e-5)


def test_spectral_kernel_eval_consistency():
    """kernel_eval must be consistent with the training spectral basis."""
    torch.manual_seed(42)
    n = 20
    m = 8
    de = 4
    e_train = torch.randn(n, de)
    e_query = torch.randn(m, de)
    from laker.kernels import SpectralAttentionKernelOperator

    op = SpectralAttentionKernelOperator(e_train, lambda_reg=0.1, num_knots=3)

    # K(query, train) from kernel_eval
    k_eval = op.kernel_eval(e_query, e_train)

    # Same via explicit projection
    cx = (e_query @ op.vh.T) * op.sigma_inv.unsqueeze(0)
    cy = (e_train @ op.vh.T) * op.sigma_inv.unsqueeze(0)
    k_manual = (cx * op.spectrum.unsqueeze(0)) @ cy.T

    torch.testing.assert_close(k_eval, k_manual, rtol=1e-4, atol=1e-5)


def test_spectral_kernel_diagonal():
    """diagonal() must match the diagonal of the dense matrix."""
    torch.manual_seed(42)
    n = 15
    de = 3
    e = torch.randn(n, de)
    from laker.kernels import SpectralAttentionKernelOperator

    op = SpectralAttentionKernelOperator(e, lambda_reg=0.2, num_knots=4)
    diag_op = op.diagonal()
    diag_dense = op.to_dense().diagonal()
    torch.testing.assert_close(diag_op, diag_dense, rtol=1e-5, atol=1e-6)


def test_spectral_shaper_monotonicity():
    """MonotoneSpectrumShaper output must be monotonic in its input."""
    from laker.kernels import MonotoneSpectrumShaper

    shaper = MonotoneSpectrumShaper(num_knots=5)
    shaper.set_knots(0.0, 10.0)
    x = torch.linspace(-2.0, 12.0, 100)
    y = shaper(x)
    diffs = torch.diff(y)
    # All differences should be non-negative (monotonic)
    assert torch.all(diffs >= -1e-6)
