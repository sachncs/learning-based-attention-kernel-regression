"""Embedding modules that map spatial locations to learned feature vectors."""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from laker.backend import get_default_device, get_default_dtype

logger = logging.getLogger(__name__)


class PositionEmbedding(nn.Module):
    """Deterministic position-driven embedding used in the LAKER paper.

    Maps each spatial coordinate vector ``x \\in R^{dx}`` to a ``de``-dimensional
    feature vector via a fixed non-linear transformation (random Fourier features
    followed by a deterministic MLP). This produces the embeddings ``E`` that
    induce the attention kernel ``G = exp(E E^T)``.

    Args:
        input_dim: Spatial dimension ``dx`` (e.g. 2 for 2-D cartography).
        embedding_dim: Embedding dimension ``de`` (paper uses 10).
        num_fourier: Number of random Fourier frequencies. If ``None``,
            defaults to ``embedding_dim * 2``.
        sigma: Bandwidth for random Fourier features.
        seed: Random seed for reproducible Fourier frequencies.
        device: torch device.
        dtype: torch dtype.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_fourier: Optional[int] = None,
        sigma: float = 10.0,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.sigma = sigma
        if num_fourier is None:
            num_fourier = embedding_dim * 2
        self.num_fourier = num_fourier

        if device is None:
            device = get_default_device()
        if dtype is None:
            dtype = get_default_dtype()

        gen = torch.Generator(device=device).manual_seed(seed)
        self.register_buffer(
            "freq",
            torch.randn(input_dim, num_fourier, generator=gen, device=device, dtype=dtype) / sigma,
        )
        self.register_buffer(
            "phase",
            torch.rand(num_fourier, generator=gen, device=device, dtype=dtype) * 2.0 * math.pi,
        )

        # Small deterministic MLP: num_fourier -> embedding_dim
        mlp_hidden = max(embedding_dim, num_fourier // 2)
        self.mlp = nn.Sequential(
            nn.Linear(num_fourier, mlp_hidden, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(mlp_hidden, embedding_dim, device=device, dtype=dtype),
        )

        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map spatial locations to embeddings.

        Args:
            x: Tensor of shape ``(n, input_dim)`` with spatial coordinates.

        Returns:
            Tensor of shape ``(n, embedding_dim)``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Random Fourier features: cos(2*pi * (x @ freq) + phase)
        features = torch.cos(2.0 * math.pi * (x @ self.freq) + self.phase)
        return self.mlp(features)

    def extra_repr(self) -> str:
        """Return a string representation of module hyperparameters."""
        return (
            f"input_dim={self.input_dim}, embedding_dim={self.embedding_dim}, "
            f"num_fourier={self.num_fourier}, sigma={self.sigma}"
        )
