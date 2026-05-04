"""Tests for reliability, determinism, and edge cases."""

import tempfile

import torch
from custom_embed import CustomEmbedding

from laker.embeddings import PositionEmbedding
from laker.models import LAKERRegressor


def test_position_embedding_determinism():
    """PositionEmbedding with same seed should produce identical weights."""
    emb1 = PositionEmbedding(input_dim=2, embedding_dim=10, seed=42, dtype=torch.float64)
    emb2 = PositionEmbedding(input_dim=2, embedding_dim=10, seed=42, dtype=torch.float64)

    x = torch.randn(5, 2, dtype=torch.float64)
    with torch.no_grad():
        out1 = emb1(x)
        out2 = emb2(x)

    torch.testing.assert_close(out1, out2)
    # MLP weights should also match
    for p1, p2 in zip(emb1.mlp.parameters(), emb2.mlp.parameters()):
        torch.testing.assert_close(p1, p2)


def test_bfloat16_mixed_precision():
    """Mixed-precision with bfloat16 should work."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float32) * 100.0
    y = torch.randn(n, dtype=torch.float32)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        embedding_dtype=torch.bfloat16,
        dtype=torch.float32,
        verbose=False,
    )
    model.fit(x, y)
    assert model.embeddings.dtype == torch.float32
    assert model.alpha is not None


def test_condition_number():
    """condition_number should return a positive finite value."""
    torch.manual_seed(42)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)
    kappa = model.condition_number()
    assert 1.0 < kappa < float("inf")


def test_custom_embedding_save_load():
    """Save/load should preserve custom embedding modules."""
    torch.manual_seed(42)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=30,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        embedding_module=CustomEmbedding(),
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    model.save(path)

    loaded = LAKERRegressor.load(path)
    assert loaded.alpha is not None
    # Prediction should match original
    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    with torch.no_grad():
        pred_orig = model.predict(x_test)
        pred_loaded = loaded.predict(x_test)
    torch.testing.assert_close(pred_orig, pred_loaded, rtol=1e-5, atol=1e-6)


def test_low_rank_save_load():
    """Save/load should preserve kernel_approx type."""
    torch.manual_seed(42)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=30,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        kernel_approx="nystrom",
        num_landmarks=30,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    model.save(path)

    loaded = LAKERRegressor.load(path)
    assert loaded.kernel_approx == "nystrom"
    assert loaded.kernel_operator.__class__.__name__ == "NystromAttentionKernelOperator"
