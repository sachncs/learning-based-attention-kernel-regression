"""Embedding modules that map spatial locations to learned feature vectors."""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from laker.backend import get_default_device, get_default_dtype

logger = logging.getLogger(__name__)


class PositionEmbedding(nn.Module):
    r"""Deterministic position-driven embedding used in the LAKER paper.

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

    Warning:
        This module temporarily manipulates the **global** PyTorch RNG state
        (``torch.manual_seed``) during initialisation to make the MLP weight
        init deterministic.  Concurrent initialisation from multiple threads
        can therefore produce non-deterministic results.  Instantiate
        ``PositionEmbedding`` objects sequentially, or pass a pre-initialised
        ``embedding_module`` to ``LAKERRegressor`` if thread safety is required.

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
        """Initialise the position embedding module."""
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

        # 1. Fourier frequencies and phases are drawn from a dedicated Generator
        #    so they are independent of the global PyTorch RNG.
        gen = torch.Generator(device=device).manual_seed(seed)
        self.register_buffer(
            "freq",
            torch.randn(
                input_dim,
                num_fourier,
                generator=gen,
                device=device,
                dtype=dtype,
            )
            / sigma,
        )
        self.register_buffer(
            "phase",
            torch.rand(num_fourier, generator=gen, device=device, dtype=dtype) * 2.0 * math.pi,
        )

        # 2. MLP weights are initialised manually with a local Generator so the
        #    global RNG is never mutated (thread-safe and deterministic).
        mlp_hidden = max(embedding_dim, num_fourier // 2)
        self.mlp = nn.Sequential(
            nn.Linear(num_fourier, mlp_hidden, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(mlp_hidden, embedding_dim, device=device, dtype=dtype),
        )
        with torch.no_grad():
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    # kaiming_uniform_-style init using local Generator
                    bound = math.sqrt(1.0 / layer.weight.shape[1])
                    layer.weight.copy_(
                        torch.rand(
                            layer.weight.shape,
                            generator=gen,
                            device=device,
                            dtype=dtype,
                        )
                        * (2 * bound)
                        - bound
                    )
                    if layer.bias is not None:
                        layer.bias.copy_(
                            torch.rand(
                                layer.bias.shape,
                                generator=gen,
                                device=device,
                                dtype=dtype,
                            )
                            * (2 * bound)
                            - bound
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
