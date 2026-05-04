"""Tests for laker.embeddings module."""

import torch

from laker.embeddings import PositionEmbedding


def test_position_embedding_forward_2d():
    """PositionEmbedding should map 2-D coordinates to embeddings."""
    embed = PositionEmbedding(input_dim=2, embedding_dim=8, seed=42)
    x = torch.rand(10, 2)
    out = embed(x)
    assert out.shape == (10, 8)


def test_position_embedding_forward_1d():
    """PositionEmbedding should handle 1-D input by unsqueezing."""
    embed = PositionEmbedding(input_dim=2, embedding_dim=8, seed=42)
    x = torch.rand(2)
    out = embed(x)
    assert out.shape == (1, 8)


def test_position_embedding_extra_repr():
    """extra_repr should return a non-empty string with key parameters."""
    embed = PositionEmbedding(input_dim=2, embedding_dim=8, num_fourier=16, sigma=5.0, seed=42)
    repr_str = embed.extra_repr()
    assert "input_dim=2" in repr_str
    assert "embedding_dim=8" in repr_str
    assert "num_fourier=16" in repr_str
    assert "sigma=5.0" in repr_str


def test_position_embedding_default_num_fourier():
    """PositionEmbedding should default num_fourier to embedding_dim * 2."""
    embed = PositionEmbedding(input_dim=2, embedding_dim=10, seed=42)
    assert embed.num_fourier == 20


def test_position_embedding_reproducibility():
    """Same seed should produce identical embeddings."""
    embed1 = PositionEmbedding(input_dim=2, embedding_dim=8, seed=42)
    embed2 = PositionEmbedding(input_dim=2, embedding_dim=8, seed=42)
    x = torch.rand(10, 2)
    out1 = embed1(x)
    out2 = embed2(x)
    torch.testing.assert_close(out1, out2)


def test_position_embedding_device_dtype():
    """PositionEmbedding should respect explicit device and dtype."""
    embed = PositionEmbedding(
        input_dim=2,
        embedding_dim=4,
        seed=42,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    x = torch.rand(5, 2, dtype=torch.float64)
    out = embed(x)
    assert out.dtype == torch.float64
    assert out.device == torch.device("cpu")
