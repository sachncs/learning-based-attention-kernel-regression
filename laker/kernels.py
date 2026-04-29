"""Attention kernel operators for matrix-free linear algebra."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class AttentionKernelOperator:
    """Matrix-free operator for the exponential attention kernel ``G = exp(E E^T)``.

    The exponential is applied **element-wise**. For large ``n`` the dense matrix
    is never materialised; instead matvecs are computed in optionally chunked
    blocks to respect GPU memory limits.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda`` added to the diagonal.
        chunk_size: If ``None``, the full kernel block is formed for matvecs.
            If an integer, chunked evaluation is used so that peak memory is
            ``O(chunk_size * n)``.
        device: torch device (inferred from ``embeddings`` if omitted).
        dtype: torch dtype (inferred from ``embeddings`` if omitted).

    Raises:
        ValueError: If ``embeddings`` is not 2-D.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        chunk_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)
        self.chunk_size = chunk_size

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype

        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        self.shape = (self.n, self.n)
        self.lambda_vec = None  # cached diagonal scaling for Jacobi preconditioning

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G)`` to vector(s) ``x``.

        Automatically dispatches to the 1-D or 2-D implementation based on
        the dimensionality of ``x``.

        Args:
            x: Tensor of shape ``(n,)`` or ``(n, k)``.

        Returns:
            Tensor of the same shape as ``x``.

        Raises:
            ValueError: If ``x`` is not 1-D or 2-D.
        """
        if x.dim() == 1:
            return self._matvec_impl(x)
        if x.dim() == 2:
            return self._matvec_impl(x)
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")

    def _matvec_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Shared implementation for 1-D and 2-D matvecs.

        Args:
            x: Tensor of shape ``(n,)`` or ``(n, k)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        out = self.lambda_reg * x
        if self.chunk_size is None or self.n <= self.chunk_size:
            gram = self.embeddings @ self.embeddings.T
            kernel = torch.exp(gram)
            out = out + kernel @ x
        else:
            for start in range(0, self.n, self.chunk_size):
                end = min(start + self.chunk_size, self.n)
                gram_chunk = self.embeddings[start:end] @ self.embeddings.T
                kernel_chunk = torch.exp(gram_chunk)
                out[start:end] = out[start:end] + kernel_chunk @ x
        return out

    def matvec_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G)`` to a single vector.

        Args:
            x: Tensor of shape ``(n,)``.

        Returns:
            Tensor of shape ``(n,)``.
        """
        return self._matvec_impl(x)

    def matvec_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G)`` to a batch of vectors.

        Args:
            x: Tensor of shape ``(n, k)``.

        Returns:
            Tensor of shape ``(n, k)``.
        """
        return self._matvec_impl(x)

    def diagonal(self) -> torch.Tensor:
        """Return the diagonal of ``lambda I + G``.

        Since ``G_{ii} = exp(||e_i||^2)``, the diagonal is
        ``lambda + exp(||e_i||^2)``.

        Returns:
            Tensor of shape ``(n,)``.
        """
        sq_norms = torch.sum(self.embeddings**2, dim=1)
        return self.lambda_reg + torch.exp(sq_norms)

    def to_dense(self) -> torch.Tensor:
        """Materialise the full dense ``(lambda I + G)`` matrix.

        Warning:
            This costs ``O(n^2)`` memory. Use only for small ``n``
            or debugging.

        Returns:
            Dense tensor of shape ``(n, n)``.
        """
        gram = self.embeddings @ self.embeddings.T
        kernel = torch.exp(gram)
        kernel.diagonal().add_(self.lambda_reg)
        return kernel

    def kernel_eval(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate the attention kernel between two sets of points.

        Args:
            x: Query embeddings of shape ``(m, embedding_dim)``.
            y: Reference embeddings of shape ``(p, embedding_dim)``. If ``None``,
                uses ``self.embeddings``.

        Returns:
            Kernel matrix of shape ``(m, p)`` or ``(m, n)``.
        """
        if y is None:
            y = self.embeddings
        gram = x @ y.T
        return torch.exp(gram)
