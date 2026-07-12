"""Embedding modules that map spatial locations to learned feature vectors.

This module contains the **position-driven embedding** used by every
LAKER component. The embeddings induce the attention kernel

.. math::

    G = \\exp(E E^\\top)

which is the central object of the regression problem. The default
:class:`PositionEmbedding` follows the construction in Tao & Tan (2026):

1. A bank of **random Fourier features** with frequencies drawn from a
   local :class:`torch.Generator` provides a deterministic shift-invariant
   basis.
2. A small two-layer MLP (Tanh activation) maps the Fourier features to
   the final ``embedding_dim``-dimensional embedding.

The MLP weights can optionally be fine-tuned end-to-end via
:meth:`laker.training.EmbeddingTrainer.fit_learned_embeddings`. When
left at their initial random values, the embedding is fully determined
by ``seed`` and the spatial coordinates.

Thread-safety and determinism are first-class concerns: the module
deliberately uses a *local* :class:`torch.Generator` for both the
Fourier frequencies and the MLP init so that concurrent instantiation
does not race on the global PyTorch RNG.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from laker.backend import get_default_device, get_default_dtype

logger = logging.getLogger(__name__)


class PositionEmbedding(nn.Module):
    r"""Deterministic position-driven embedding used in the LAKER paper.

    Maps each spatial coordinate vector ``x \in \mathbb{R}^{d_x}`` to a
    ``d_e``-dimensional feature vector via a fixed non-linear
    transformation (random Fourier features followed by a deterministic
    MLP). The output induces the attention kernel
    ``G = \exp(E E^\top)`` that defines the LAKER regression problem.

    The module is fully deterministic for a given ``seed``: the Fourier
    frequencies, phases, and MLP initialisation are all drawn from a
    dedicated :class:`torch.Generator`, never the global PyTorch RNG.

    Args:
        input_dim: Spatial dimension ``d_x`` (e.g. ``2`` for 2-D
            cartography, ``3`` for volumetric).
        embedding_dim: Output embedding dimension ``d_e``. The original
            LAKER paper uses ``10``.
        num_fourier: Number of random Fourier frequencies. If ``None``,
            defaults to ``2 * embedding_dim`` (twice the embedding
            dimension, matching the paper's settings).
        sigma: Standard deviation of the Fourier frequency distribution
            (``freq ~ N(0, 1/\sigma^2)``). A larger ``sigma`` produces
            higher-frequency features.
        seed: Random seed for reproducible Fourier frequencies and MLP
            init.
        device: :class:`torch.device`. Defaults to
            :func:`laker.backend.get_default_device`.
        dtype: :class:`torch.dtype`. Defaults to
            :func:`laker.backend.get_default_dtype`.

    Raises:
        ValueError: If ``input_dim``, ``embedding_dim``, or ``num_fourier``
            is not strictly positive (propagated from PyTorch layer
            initialisation).

    Warning:
        The init code is fully thread-safe because it relies on a local
        :class:`torch.Generator`. However, concurrent *training* of the
        MLP weights from multiple threads is not supported because the
        underlying ``nn.Linear`` parameters live on a single device and
        ``Adam``/``SGD`` optimisers are not thread-safe by default.

    Example:
        >>> embed = PositionEmbedding(input_dim=2, embedding_dim=10)
        >>> x = torch.rand(100, 2)
        >>> e = embed(x)
        >>> e.shape
        torch.Size([100, 10])

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
        """Initialise the position embedding module.

        See the class docstring for full parameter semantics. The
        constructor performs two deterministic draws from a local
        :class:`torch.Generator`:

        1. ``freq`` and ``phase`` buffers (``num_fourier`` frequencies
           per spatial dimension plus ``num_fourier`` scalar phases) for
           the Fourier feature map.
        2. The MLP weights and biases, initialised with a Kaiming-uniform
           distribution scaled by ``sqrt(1 / fan_in)`` using the same
           local generator.

        Args:
            input_dim: Spatial dimension.
            embedding_dim: Output dimension.
            num_fourier: Number of Fourier frequencies (defaults to
                ``2 * embedding_dim``).
            sigma: Fourier bandwidth.
            seed: RNG seed.
            device: Target device.
            dtype: Target dtype.

        Side effects:
            Registers two buffers (``freq``, ``phase``) and two
            :class:`torch.nn.Linear` submodules on the module.

        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.sigma = sigma
        if num_fourier is None:
            # The paper uses twice the embedding dimension as a default;
            # this gives the MLP enough capacity to mix the Fourier
            # features without redundancy.
            num_fourier = embedding_dim * 2
        self.num_fourier = num_fourier

        if device is None:
            device = get_default_device()
        if dtype is None:
            dtype = get_default_dtype()

        # ------------------------------------------------------------------
        # Fourier feature bank
        # ------------------------------------------------------------------
        # Draw frequencies and phases from a *dedicated* Generator so the
        # global PyTorch RNG is never mutated. This makes the module
        # thread-safe at construction time.
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
            # ``freq ~ N(0, 1/sigma^2)`` so the resulting Fourier features
            # have spatial scale ``~ 1/sigma``.
            / sigma,
        )
        # Phase is drawn from the *same* generator as ``freq`` so the
        # whole module is reproducible from a single ``seed``.
        self.register_buffer(
            "phase",
            torch.rand(num_fourier, generator=gen, device=device, dtype=dtype) * 2.0 * math.pi,
        )

        # ------------------------------------------------------------------
        # MLP weights
        # ------------------------------------------------------------------
        # ``mlp_hidden`` is the larger of ``embedding_dim`` and
        # ``num_fourier // 2``; this guarantees the bottleneck either
        # preserves the embedding dimension or projects down by at most
        # 2x, which empirically matches the paper's settings.
        mlp_hidden = max(embedding_dim, num_fourier // 2)
        self.mlp = nn.Sequential(
            nn.Linear(num_fourier, mlp_hidden, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(mlp_hidden, embedding_dim, device=device, dtype=dtype),
        )
        with torch.no_grad():
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    # Kaiming-uniform_-style init under the *local*
                    # Generator; this keeps the global RNG untouched and
                    # is therefore thread-safe even when many modules are
                    # constructed in parallel.
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

        Applies the random Fourier feature map ``cos(2π (x @ freq) + phase)``
        and passes the result through the two-layer MLP.

        Args:
            x: Tensor of shape ``(n, input_dim)`` with spatial
                coordinates. A 1-D tensor ``(input_dim,)`` is
                automatically unsqueezed to ``(1, input_dim)`` and the
                output is squeezed back to ``(embedding_dim,)`` so that
                single-point queries "just work".

        Returns:
            Tensor of shape ``(n, embedding_dim)`` containing the
            embeddings.

        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Random Fourier features: cos(2*pi * (x @ freq) + phase)
        features = torch.cos(2.0 * math.pi * (x @ self.freq) + self.phase)
        return self.mlp(features)

    def extra_repr(self) -> str:
        """Return a string representation of module hyperparameters.

        Used by :meth:`torch.nn.Module.__repr__` to produce a compact,
        human-readable summary (e.g. ``"input_dim=2, embedding_dim=10,
        num_fourier=20, sigma=10.0"``).

        Returns:
            A single-line string describing the constructor arguments.

        """
        return (
            f"input_dim={self.input_dim}, embedding_dim={self.embedding_dim}, "
            f"num_fourier={self.num_fourier}, sigma={self.sigma}"
        )
