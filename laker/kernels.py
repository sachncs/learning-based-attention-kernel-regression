"""Attention kernel operators for matrix-free linear algebra.

Includes exact, low-rank (Nyström, RFF), sparse k-NN, SKI, spectral-shaped, and two-scale approximations.
"""

import logging
import math
from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class KernelOperator(Protocol):
    """Protocol for matrix-free kernel operators."""

    n: int
    embedding_dim: int
    lambda_reg: float
    dtype: torch.dtype
    device: torch.device
    shape: Tuple[int, int]

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the operator to a vector."""
        ...

    def diagonal(self) -> torch.Tensor:
        """Return the diagonal of the operator."""
        ...

    def to_dense(self) -> torch.Tensor:
        """Materialise the full dense matrix."""
        ...

    def kernel_eval(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Evaluate the kernel between points."""
        ...


# ---------------------------------------------------------------------------
# Safe element-wise exponential
# ---------------------------------------------------------------------------


def exp_safe(
    gram: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    skip_clamp: bool = False,
) -> torch.Tensor:
    """Element-wise exp with dtype-aware overflow guard.

    Clamps to ``80.0`` for float32 and ``700.0`` for float64 before
    exponentiation to prevent silent overflow to ``inf``.

    When ``gram.requires_grad`` is ``True`` (e.g. during learned-embedding
    training) the ``out=`` form is skipped because PyTorch does not support
    autodiff through in-place ``torch.exp``.

    If ``skip_clamp`` is ``True`` the clamp step is bypassed.  Callers that
    pre-verify their data never exceeds the safe threshold (e.g. via
    ``max_sq_norm``) should set this to recover the original speed.
    """
    if not skip_clamp:
        max_val = 80.0 if gram.dtype == torch.float32 else 700.0
        if gram.requires_grad:
            return torch.exp(gram.clamp(max=max_val))
        if out is None:
            out = gram.clone()
        elif out is not gram:
            out.copy_(gram)
        out.clamp_(max=max_val)
        return torch.exp(out)
    # Fast path: no clamp needed
    if out is None:
        return torch.exp(gram)
    if gram.requires_grad:
        return torch.exp(gram)
    return torch.exp(gram, out=out)


# ---------------------------------------------------------------------------
# Exact attention kernel
# ---------------------------------------------------------------------------


def dense_attention_matvec(
    embeddings: torch.Tensor,
    lambda_reg: float,
    x: torch.Tensor,
    skip_clamp: bool = False,
) -> torch.Tensor:
    """Apply the non-chunked attention matvec."""
    gram = embeddings @ embeddings.T
    exp_safe(gram, out=gram, skip_clamp=skip_clamp)
    return lambda_reg * x + gram @ x


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
        """Initialise the exact attention kernel operator."""
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

        # Pre-compute whether gram values can ever overflow; if not we skip the
        # clamp in ``exp_safe`` and recover the original single-kernel speed.
        max_sq_norm = torch.sum(self.embeddings**2, dim=1).max().item()
        safe_limit = 80.0 if self.dtype == torch.float32 else 700.0
        self.skip_clamp = max_sq_norm < safe_limit

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
            return self.matvec_impl(x)
        if x.dim() == 2:
            return self.matvec_impl(x)
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")

    def matvec_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Shared implementation for 1-D and 2-D matvecs.

        Uses 2-D tiling when chunked so that peak memory is
        ``O(chunk_size^2)`` rather than ``O(chunk_size * n)``.

        Args:
            x: Tensor of shape ``(n,)`` or ``(n, k)``.

        Returns:
            Tensor of the same shape as ``x``.

        """
        if self.chunk_size is None or self.n <= self.chunk_size:
            return dense_attention_matvec(
                self.embeddings, self.lambda_reg, x, skip_clamp=self.skip_clamp
            )

        out = self.lambda_reg * x

        chunk_size_local = self.chunk_size
        n = self.n
        # Heuristic: if a single output chunk against all inputs fits comfortably
        # in memory (<= 64 MB), use fast 1-D chunking; otherwise use 2-D tiling.
        element_size = 4 if self.dtype == torch.float32 else 8
        mem_per_chunk = chunk_size_local * n * element_size
        if mem_per_chunk <= 64 * 1024 * 1024:
            for start in range(0, n, chunk_size_local):
                end = min(start + chunk_size_local, n)
                gram_chunk = self.embeddings[start:end] @ self.embeddings.T
                exp_safe(gram_chunk, out=gram_chunk, skip_clamp=self.skip_clamp)
                if x.dim() == 1:
                    out[start:end].addmv_(gram_chunk, x)
                else:
                    out[start:end].addmm_(gram_chunk, x)
            return out

        # 2-D tiling: chunk both the output and reduction dimensions
        if x.dim() == 1:
            for i_start in range(0, n, chunk_size_local):
                i_end = min(i_start + chunk_size_local, n)
                accum = torch.zeros(i_end - i_start, device=self.device, dtype=self.dtype)
                e_i = self.embeddings[i_start:i_end]
                for j_start in range(0, n, chunk_size_local):
                    j_end = min(j_start + chunk_size_local, n)
                    gram_block = e_i @ self.embeddings[j_start:j_end].T
                    exp_safe(gram_block, out=gram_block, skip_clamp=self.skip_clamp)
                    accum.addmv_(gram_block, x[j_start:j_end])
                out[i_start:i_end].add_(accum)
        else:
            k = x.shape[1]
            for i_start in range(0, n, chunk_size_local):
                i_end = min(i_start + chunk_size_local, n)
                accum = torch.zeros(i_end - i_start, k, device=self.device, dtype=self.dtype)
                e_i = self.embeddings[i_start:i_end]
                for j_start in range(0, n, chunk_size_local):
                    j_end = min(j_start + chunk_size_local, n)
                    gram_block = e_i @ self.embeddings[j_start:j_end].T
                    exp_safe(gram_block, out=gram_block, skip_clamp=self.skip_clamp)
                    accum.addmm_(gram_block, x[j_start:j_end])
                out[i_start:i_end].add_(accum)
        return out

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
        torch.exp(gram, out=gram)
        gram.diagonal().add_(self.lambda_reg)
        return gram

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate the attention kernel between two sets of points.

        Supports 2-D tiling via ``chunk_size`` so that peak memory stays
        ``O(chunk_size^2)`` even for large query/reference sets.

        Args:
            x: Query embeddings of shape ``(m, embedding_dim)``.
            y: Reference embeddings of shape ``(p, embedding_dim)``. If ``None``,
                uses ``self.embeddings``.
            chunk_size: Tile size. Defaults to ``self.chunk_size``.

        Returns:
            Kernel matrix of shape ``(m, p)`` or ``(m, n)``.

        """
        if y is None:
            y = self.embeddings
        if chunk_size is None:
            chunk_size = self.chunk_size
        m = x.shape[0]
        p = y.shape[0]
        if chunk_size is None or m <= chunk_size:
            gram = x @ y.T
            exp_safe(gram, out=gram, skip_clamp=self.skip_clamp)
            return gram

        # Use 1-D chunking over the query dimension when memory is moderate,
        # otherwise fall back to full 2-D tiling.
        element_size = 4 if self.dtype == torch.float32 else 8
        mem_per_chunk = chunk_size * p * element_size
        if mem_per_chunk <= 64 * 1024 * 1024:
            out = torch.empty(m, p, device=self.device, dtype=self.dtype)
            for start in range(0, m, chunk_size):
                end = min(start + chunk_size, m)
                gram_chunk = x[start:end] @ y.T
                exp_safe(gram_chunk, out=gram_chunk, skip_clamp=self.skip_clamp)
                out[start:end] = gram_chunk
            return out

        out = torch.empty(m, p, device=self.device, dtype=self.dtype)
        for i_start in range(0, m, chunk_size):
            i_end = min(i_start + chunk_size, m)
            for j_start in range(0, p, chunk_size):
                j_end = min(j_start + chunk_size, p)
                gram_block = x[i_start:i_end] @ y[j_start:j_end].T
                exp_safe(gram_block, out=gram_block, skip_clamp=self.skip_clamp)
                out[i_start:i_end, j_start:j_end] = gram_block
        return out


# ---------------------------------------------------------------------------
# Low-rank approximations
# ---------------------------------------------------------------------------


class NystromAttentionKernelOperator:
    r"""Nyström low-rank approximation of the exponential attention kernel.

    Approximates ``G = exp(E E^T)`` using ``m`` landmark points:

    .. math::
        G \\approx G_{nm} G_{mm}^{-1} G_{nm}^T

    where ``G_{nm}`` is the kernel between all ``n`` points and ``m`` landmarks,
    and ``G_{mm}`` is the kernel among landmarks. This reduces matvec cost
    from ``O(n^2)`` to ``O(n*m)``.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        num_landmarks: Number of landmark points ``m``. If ``None``, defaults to
            ``max(50, int(sqrt(n)))``.
        chunk_size: Chunk size for landmark kernel evaluation.
        device: torch device.
        dtype: torch dtype.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.

    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        num_landmarks: Optional[int] = None,
        landmark_method: str = "greedy",
        landmark_pilot_size: int = 1000,
        chunk_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise the Nyström kernel operator."""
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)
        self.chunk_size = chunk_size
        self.landmark_method = landmark_method
        self.landmark_pilot_size = landmark_pilot_size

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype

        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        m = num_landmarks if num_landmarks is not None else max(50, int(self.n**0.5))
        self.m = min(m, self.n)

        # Landmark norms bounded by training embedding norms — skip clamp.
        self.skip_clamp = True

        self.landmark_indices = self.select_landmarks()
        self.landmark_embeddings = self.embeddings[self.landmark_indices]

        # Compute K_nm (n, m) and K_mm (m, m)
        self.k_nm = self.compute_kernel_matrix(self.embeddings, self.landmark_embeddings)
        self.k_mm = self.compute_kernel_matrix(self.landmark_embeddings, self.landmark_embeddings)

        # Regularised Cholesky of K_mm for stable solves
        k_mm_reg = self.k_mm + 1e-6 * torch.eye(self.m, device=device, dtype=dtype)
        self.k_mm_chol = torch.linalg.cholesky(k_mm_reg)

        # Precompute K_nm @ K_mm^{-1} for fast matvecs via Cholesky solve
        self.k_nm_kmm_inv = torch.linalg.solve_triangular(
            self.k_mm_chol.T,
            torch.linalg.solve_triangular(self.k_mm_chol, self.k_nm.T, upper=False),
            upper=True,
        ).T  # (n, m)

        self.shape = (self.n, self.n)

    def select_landmarks(self) -> torch.Tensor:
        """Select landmark indices.

        Supports ``"greedy"`` (k-means++ style farthest-first) and
        ``"leverage"`` (ridge leverage score sampling from a pilot kernel).
        """
        if self.landmark_method == "greedy":
            return self._select_landmarks_greedy()
        if self.landmark_method == "leverage":
            return self._select_landmarks_leverage()
        raise ValueError(f"Unknown landmark_method={self.landmark_method}")

    def _select_landmarks_greedy(self) -> torch.Tensor:
        """Greedy landmark selection (k-means++ style)."""
        indices = torch.zeros(self.m, dtype=torch.long, device=self.device)
        indices[0] = torch.randint(0, self.n, (1,), device=self.device)

        for i in range(1, self.m):
            selected = self.embeddings[indices[:i]]
            dists = torch.cdist(self.embeddings, selected) ** 2
            min_dists = dists.min(dim=1).values
            indices[i] = min_dists.argmax()

        return indices

    def _select_landmarks_leverage(self) -> torch.Tensor:
        """Ridge leverage score sampling via pilot kernel eigendecomposition."""
        pilot_size = min(self.landmark_pilot_size, self.n)
        if pilot_size == self.n:
            pilot_idx = torch.arange(self.n, device=self.device)
        else:
            pilot_idx = torch.randperm(self.n, device=self.device)[:pilot_size]

        pilot_embeddings = self.embeddings[pilot_idx]
        gram = pilot_embeddings @ pilot_embeddings.T
        k_pilot = exp_safe(gram, skip_clamp=self.skip_clamp)

        # Eigendecomposition: K = U Λ U^T
        eigenvalues, eigenvectors = torch.linalg.eigh(k_pilot)
        # Leverage scores: l_i = Σ_k (λ_k / (λ_k + λ)) * U_{ik}^2
        scaled = eigenvalues / (eigenvalues + self.lambda_reg)
        leverage_scores = (eigenvectors**2) @ scaled  # (pilot_size,)
        leverage_scores = leverage_scores.clamp(min=0)

        # Sample m landmarks without replacement proportional to scores
        probs = leverage_scores / leverage_scores.sum()
        sampled = torch.multinomial(probs, self.m, replacement=False)
        return pilot_idx[sampled]

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute exponential attention kernel matrix."""
        gram = x @ y.T
        return exp_safe(gram, skip_clamp=self.skip_clamp)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G_approx)`` to vector(s)."""
        out = self.lambda_reg * x
        if x.dim() == 1:
            # G @ x = K_nm @ K_mm^{-1} @ K_nm^T @ x
            temp = self.k_nm_kmm_inv.T @ x  # (m,)
            out = out + self.k_nm_kmm_inv @ temp
        else:
            temp = self.k_nm_kmm_inv.T @ x  # (m, k)
            out = out + self.k_nm_kmm_inv @ temp
        return out

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G_approx``."""
        # Diagonal of G_approx = sum over j of (K_nm @ K_mm^{-1})_{ij} * (K_nm)_{ij}
        diag_approx = torch.sum(self.k_nm_kmm_inv * self.k_nm, dim=1)
        return self.lambda_reg + diag_approx

    def to_dense(self) -> torch.Tensor:
        """Materialise full approximate kernel matrix."""
        k_approx = self.k_nm_kmm_inv @ self.k_nm.T
        k_approx.diagonal().add_(self.lambda_reg)
        return k_approx

    def kernel_eval(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Evaluate kernel between queries and training points."""
        if y is None:
            y = self.embeddings
        gram = x @ y.T
        return exp_safe(gram)


class RandomFeatureAttentionKernelOperator:
    """Random Fourier Feature (RFF) approximation of the exponential kernel.

    Uses random Fourier features to approximate the Gaussian-like kernel
    induced by the exponential of inner products. For embeddings with bounded
    norm, the exponential kernel is approximated by a finite-dimensional
    feature map.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        num_features: Number of random Fourier features. If ``None``, defaults
            to ``max(100, int(sqrt(n) * 2))``.
        sigma: Bandwidth for random Fourier features.
        device: torch device.
        dtype: torch dtype.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.

    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        num_features: Optional[int] = None,
        sigma: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise the random-feature kernel operator."""
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)
        self.sigma = float(sigma)

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype

        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        r = num_features if num_features is not None else max(100, int(self.n**0.5 * 2))
        self.num_features = r

        # Generate random Fourier frequencies
        gen = torch.Generator(device=device).manual_seed(42)
        self.freq = (
            torch.randn(self.embedding_dim, r, generator=gen, device=device, dtype=dtype) / sigma
        )
        self.phase = torch.rand(r, generator=gen, device=device, dtype=dtype) * 2.0 * math.pi

        # Compute feature map Phi: (n, 2*r) [cos, sin]
        proj = self.embeddings @ self.freq
        phi = torch.cat([torch.cos(proj + self.phase), torch.sin(proj + self.phase)], dim=1)
        # Normalise so that Phi @ Phi.T approximates the kernel
        self.phi = phi / (r**0.5)

        self.shape = (self.n, self.n)
        # RFF features are bounded by sqrt(r) per component, so overflow is
        # impossible for realistic embedding norms — skip clamping.
        self.skip_clamp = True

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G_approx)`` to vector(s)."""
        out = self.lambda_reg * x
        if x.dim() == 1:
            temp = self.phi.T @ x  # (2*r,)
            out = out + self.phi @ temp
        else:
            temp = self.phi.T @ x  # (2*r, k)
            out = out + self.phi @ temp
        return out

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G_approx``."""
        sq_norms = torch.sum(self.phi**2, dim=1)
        return self.lambda_reg + sq_norms

    def to_dense(self) -> torch.Tensor:
        """Materialise full approximate kernel matrix."""
        k_approx = self.phi @ self.phi.T
        k_approx.diagonal().add_(self.lambda_reg)
        return k_approx

    def kernel_eval(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Evaluate approximate kernel between queries and training points."""
        if y is None:
            y = self.embeddings
        # Build RFF feature maps for both sets using the same random frequencies
        proj_x = x @ self.freq
        phi_x = torch.cat(
            [torch.cos(proj_x + self.phase), torch.sin(proj_x + self.phase)],
            dim=1,
        )
        proj_y = y @ self.freq
        phi_y = torch.cat(
            [torch.cos(proj_y + self.phase), torch.sin(proj_y + self.phase)],
            dim=1,
        )
        return (phi_x @ phi_y.T) / self.num_features


# ---------------------------------------------------------------------------
# Sparse k-NN approximation
# ---------------------------------------------------------------------------


class SparseKNNAttentionKernelOperator:
    """Sparse k-NN approximation of the exponential attention kernel.

    For each point, retains only the ``k`` largest kernel values (nearest
    neighbours in inner-product space).  This reduces storage from ``O(n^2)``
    to ``O(n*k)`` and matvec cost from ``O(n^2)`` to ``O(n*k)``.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        k_neighbors: Number of neighbours to retain.  If ``None``, defaults
            to ``min(50, n)``.
        chunk_size: Chunk size for distance computations.
        device: torch device.
        dtype: torch dtype.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.

    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        k_neighbors: Optional[int] = None,
        chunk_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise the sparse k-NN kernel operator."""
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

        k = k_neighbors if k_neighbors is not None else min(50, self.n)
        self.k_neighbors = min(k, self.n)

        self.build_sparse_knn()
        # Cache the coalesced sparse matrix once — rebuilding on every matvec
        # would incur O(n^2) coalescing overhead that dominates the actual SpMV.
        self.cached_sparse_mat = self.sparse_mat(self.n, self.n)
        self.shape = (self.n, self.n)
        # Sparse KNN selects neighbours by Euclidean distance in embedding space,
        # then computes exp of inner products for the selected pairs only.
        # These values are bounded by the k-nearest distances, so overflow is
        # extremely unlikely — skip clamping for consistency with other kernels.
        self.skip_clamp = True

    def build_sparse_knn(self) -> None:
        """Compute top-k Euclidean neighbours, symmetrise, and store as COO."""
        n = self.n
        k = self.k_neighbors
        chunk_size_local = self.chunk_size or n

        row_list = []
        col_list = []
        val_list = []

        for i_start in range(0, n, chunk_size_local):
            i_end = min(i_start + chunk_size_local, n)
            # Euclidean distance in embedding space; self is always distance 0
            dists = torch.cdist(
                self.embeddings[i_start:i_end], self.embeddings
            )  # (chunk_size_local, n)
            # k-1 nearest neighbours + self (distance 0)
            topk = torch.topk(dists, min(k, n), largest=False, dim=1)
            row_idx = (
                torch.arange(i_start, i_end, device=self.device).unsqueeze(1).expand(-1, min(k, n))
            )
            row_list.append(row_idx.flatten())
            col_list.append(topk.indices.flatten())
            # Compute exact kernel values for the selected pairs
            gram_vals = torch.sum(
                self.embeddings[i_start:i_end].unsqueeze(1) * self.embeddings[topk.indices],
                dim=2,
            ).flatten()
            val_list.append(exp_safe(gram_vals))

        rows = torch.cat(row_list)
        cols = torch.cat(col_list)
        vals = torch.cat(val_list)

        # Symmetrise: if (i,j) is an edge, ensure (j,i) is also present.
        all_rows = torch.cat([rows, cols])
        all_cols = torch.cat([cols, rows])
        all_vals = torch.cat([vals, vals])

        # Deduplicate by sorting on a composite key
        sort_key = all_rows.to(torch.int64) * (n + 1) + all_cols.to(torch.int64)
        order = torch.argsort(sort_key)
        sorted_rows = all_rows[order]
        sorted_cols = all_cols[order]
        sorted_vals = all_vals[order]

        diff = torch.diff(sorted_rows.to(torch.int64) * (n + 1) + sorted_cols.to(torch.int64))
        is_new = torch.cat([torch.tensor([True], device=self.device), diff != 0])

        coo_indices = torch.stack([sorted_rows[is_new], sorted_cols[is_new]], dim=0)
        coo_values = sorted_vals[is_new]

        # Ensure every row has a diagonal entry and enforce strict diagonal
        # dominance so the symmetrised matrix is guaranteed positive definite.
        # When k >= n the matrix is already exact and PSD, so skip this step.
        if k < n:
            diag_mask = coo_indices[0] == coo_indices[1]
            diag_rows = coo_indices[0, diag_mask]
            diag_vals = coo_values[diag_mask]

            off_mask = ~diag_mask
            off_rows = coo_indices[0, off_mask]
            off_vals = coo_values[off_mask]

            row_sums = torch.zeros(n, device=self.device, dtype=self.dtype)
            row_sums.index_add_(0, off_rows, off_vals.abs())

            diag_map = torch.full((n,), float("-inf"), device=self.device, dtype=self.dtype)
            diag_map[diag_rows] = diag_vals

            min_diag = row_sums * 1.01 + 1e-8
            new_diag = torch.maximum(diag_map, min_diag)

            # Update existing diagonals
            coo_values = coo_values.clone()
            coo_values[diag_mask] = new_diag[diag_rows]

            # Add missing diagonal entries as new COO elements
            missing = torch.where(~torch.isfinite(diag_map))[0]
            if missing.numel() > 0:
                missing_vals = min_diag[missing]
                coo_indices = torch.cat(
                    [coo_indices, torch.stack([missing, missing], dim=0)], dim=1
                )
                coo_values = torch.cat([coo_values, missing_vals])

        self.coo_indices = coo_indices
        self.coo_values = coo_values

    def sparse_mat(self, m: int, n: int) -> torch.Tensor:
        """Build a sparse COO tensor of shape (m, n)."""
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            return torch.sparse_coo_tensor(
                self.coo_indices,
                self.coo_values,
                (m, n),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G_approx)`` to vector(s)."""
        out = self.lambda_reg * x
        if x.dim() == 1:
            out = out + torch.sparse.mm(self.cached_sparse_mat, x.unsqueeze(1)).squeeze(1)
        else:
            out = out + torch.sparse.mm(self.cached_sparse_mat, x)
        return out

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G_approx``."""
        diag = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        diag_mask = self.coo_indices[0] == self.coo_indices[1]
        diag_rows = self.coo_indices[0, diag_mask]
        diag_vals = self.coo_values[diag_mask]
        diag[diag_rows] = diag_vals
        # Ensure any missing diagonals are at least the exact kernel diagonal
        sq_norms = torch.sum(self.embeddings**2, dim=1)
        exact_diag = torch.exp(sq_norms)
        diag = torch.maximum(diag, exact_diag)
        return self.lambda_reg + diag

    def to_dense(self) -> torch.Tensor:
        """Materialise full dense matrix (for debugging only)."""
        dense = self.sparse_mat(self.n, self.n).to_dense()
        dense.diagonal().add_(self.lambda_reg)
        return dense

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate sparse kernel between queries and training points.

        Returns a **sparse** tensor of shape ``(m, p)`` where each row
        contains the top-``k`` kernel values.  This enables memory-efficient
        ``predict()`` via ``torch.sparse.mm``.
        """
        if y is None:
            y = self.embeddings
        m = x.shape[0]
        p = y.shape[0]
        k = min(self.k_neighbors, p)
        chunk_size_local = chunk_size or m

        row_list = []
        col_list = []
        val_list = []

        for i_start in range(0, m, chunk_size_local):
            i_end = min(i_start + chunk_size_local, m)
            gram_chunk = x[i_start:i_end] @ y.T  # (chunk_size_local, p)
            topk = torch.topk(gram_chunk, k, largest=True, dim=1)
            row_idx = torch.arange(i_start, i_end, device=self.device).unsqueeze(1).expand(-1, k)
            row_list.append(row_idx.flatten())
            col_list.append(topk.indices.flatten())
            val_list.append(exp_safe(topk.values).flatten())

        rows = torch.cat(row_list)
        cols = torch.cat(col_list)
        vals = torch.cat(val_list)

        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            return torch.sparse_coo_tensor(
                torch.stack([rows, cols], dim=0),
                vals,
                (m, p),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()


# ---------------------------------------------------------------------------
# SKI approximation
# ---------------------------------------------------------------------------


def multilinear_weights(
    x: torch.Tensor, grid_1d: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multilinear interpolation weights for a product grid.

    Args:
        x: Tensor of shape ``(n, d)`` with coordinates in [0, 1] per dim.
        grid_1d: List of ``d`` tensors, each of shape ``(g_i,)`` with the
            1-D grid coordinates (sorted ascending).

    Returns:
        indices: Long tensor of shape ``(n, num_vertices)`` with grid-point
            linear indices.
        weights: Tensor of shape ``(n, num_vertices)`` with interpolation
            weights (sum to 1 per row).

    """
    n, d = x.shape
    g_per_dim = [g.shape[0] for g in grid_1d]
    vertices = 2**d

    # For each dimension find the lower bin index and fractional offset
    low_idx = torch.zeros(n, d, dtype=torch.long, device=x.device)
    frac = torch.zeros(n, d, dtype=x.dtype, device=x.device)
    for dim, g in enumerate(grid_1d):
        g = g.to(x.device, x.dtype)
        # Clamp x to grid bounds
        xc = x[:, dim].clamp(min=g[0], max=g[-1])
        # Find the rightmost grid point <= xc (searchsorted is not on all PyTorch versions)
        # Use broadcasting approach
        diff = xc.unsqueeze(1) - g.unsqueeze(0)  # (n, g_i)
        # Find last non-positive difference
        # For values exactly at grid points, this gives the next index; handle separately
        idx = (diff > 0).sum(dim=1) - 1
        idx = idx.clamp(min=0, max=g.shape[0] - 2)
        low = g[idx]
        high = g[idx + 1]
        denom = high - low
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        low_idx[:, dim] = idx
        frac[:, dim] = (xc - low) / denom

    # Build all 2^d vertex combinations
    vertex_offsets = torch.arange(vertices, device=x.device)
    bits = ((vertex_offsets.unsqueeze(1) >> torch.arange(d, device=x.device)) & 1).to(torch.bool)

    # Grid strides for linear index
    strides = [1]
    for g in reversed(g_per_dim[1:]):
        strides.append(strides[-1] * g)
    strides = list(reversed(strides))
    strides_t = torch.tensor(strides, dtype=torch.long, device=x.device)

    indices = torch.zeros(n, vertices, dtype=torch.long, device=x.device)
    weights = torch.ones(n, vertices, dtype=x.dtype, device=x.device)
    for dim in range(d):
        dim_idx = low_idx[:, dim].unsqueeze(1) + bits[:, dim].unsqueeze(0).long()  # (n, vertices)
        indices += dim_idx * strides_t[dim]
        dim_weight = torch.where(
            bits[:, dim].unsqueeze(0),
            frac[:, dim].unsqueeze(1),
            1.0 - frac[:, dim].unsqueeze(1),
        )
        weights *= dim_weight

    return indices, weights


class SKIAttentionKernelOperator:
    """SKI approximation of the exponential attention kernel.

    Builds a regular product grid in the embedding space and uses multilinear
    interpolation weights ``W`` so that ``K ≈ W K_grid W^T``.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        grid_size: Maximum number of grid points.  The actual grid is a
            product grid with ``floor(grid_size**(1/d))`` points per
            dimension, capped so the product does not exceed ``grid_size``.
        grid_bounds: Optional ``(d, 2)`` tensor with ``[min, max]`` per
            dimension.  If ``None``, inferred from data.
        device: torch device.
        dtype: torch dtype.

    Raises:
        ValueError: If ``embeddings`` is not 2-D or ``grid_size`` is too small.

    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        grid_size: Optional[int] = None,
        grid_bounds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise the SKI kernel operator."""
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype

        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        d = self.embedding_dim
        if grid_size is None:
            grid_size = min(4096, max(64, 2**d))
        self.grid_size = grid_size

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")

        # Determine points per dimension for product grid
        grid_points_per_dimension = max(2, int(grid_size ** (1.0 / d)))
        while grid_points_per_dimension**d > grid_size and grid_points_per_dimension > 2:
            grid_points_per_dimension -= 1
        self.grid_points_per_dim = grid_points_per_dimension
        actual_grid = grid_points_per_dimension**d
        if actual_grid > grid_size:
            raise ValueError(
                f"Cannot build product grid: {grid_points_per_dimension}^{d}={actual_grid} "
                f"> {grid_size}. Use a larger grid_size or lower embedding_dim."
            )

        if self.dtype == torch.float32 and actual_grid > 8192:
            logger.warning(
                "SKI grid has %d points; exact kernel evaluation on the grid may be slow. "
                "Consider reducing grid_size for embedding_dim=%d.",
                actual_grid,
                d,
            )

        self.build_grid(grid_bounds)
        self.shape = (self.n, self.n)
        # SKI grid points are drawn from the same bounded embedding space as
        # the training data — exp of their inner products cannot overflow.
        self.skip_clamp = True

    def build_grid(self, grid_bounds: Optional[torch.Tensor]) -> None:
        """Construct product grid and interpolation weights."""
        d = self.embedding_dim
        grid_points_per_dimension = self.grid_points_per_dim

        if grid_bounds is None:
            mins = self.embeddings.min(dim=0).values
            maxs = self.embeddings.max(dim=0).values
            # Add small padding
            pad = (maxs - mins) * 0.05 + 1e-6
            mins = mins - pad
            maxs = maxs + pad
        else:
            mins = grid_bounds[:, 0]
            maxs = grid_bounds[:, 1]

        # 1-D grids per dimension
        grid_1d = [
            torch.linspace(
                mins[i].item(),
                maxs[i].item(),
                grid_points_per_dimension,
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(d)
        ]
        self.grid_1d = grid_1d

        # Normalise embeddings to [0, 1] for interpolation
        norm_embed = torch.zeros_like(self.embeddings)
        for i in range(d):
            denom = maxs[i] - mins[i]
            denom = denom if denom > 0 else 1.0
            norm_embed[:, i] = (self.embeddings[:, i] - mins[i]) / denom

        # Normalise grid to [0, 1] as well (so grid_1d_norm[i][j] = j/(gpd-1))
        grid_1d_norm = [
            torch.linspace(
                0.0,
                1.0,
                grid_points_per_dimension,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(d)
        ]

        indices, weights = multilinear_weights(norm_embed, grid_1d_norm)
        self.interp_indices = indices  # (n, vertices)
        self.interp_weights = weights  # (n, vertices)

        # Build full product grid coordinates
        mesh = torch.meshgrid(*grid_1d, indexing="ij")
        grid_points = torch.stack([m.flatten() for m in mesh], dim=1).to(
            self.dtype
        )  # (actual_grid, d)
        self.grid_points = grid_points

        gram_grid = grid_points @ grid_points.T
        self.k_grid = exp_safe(gram_grid)  # (actual_grid, actual_grid)

        # Precompute W @ K_grid for fast matvec: W @ K_grid @ (W^T @ x)
        # W is sparse n x actual_grid; we store it as indices + weights
        # For matvec we compute v = W^T @ x, then u = K_grid @ v, then W @ u
        # W^T @ x can be done with index_add

    def weights_x(self, x: torch.Tensor) -> torch.Tensor:
        """Compute W^T @ x efficiently via index_add."""
        g = self.grid_points.shape[0]
        out = torch.zeros(g, *x.shape[1:], device=self.device, dtype=self.dtype)
        # For each data point i, distribute x[i] * weight[i,j] to grid point interp_indices[i,j]
        vertices = self.interp_indices.shape[1]
        for v in range(vertices):
            idx = self.interp_indices[:, v]
            w = self.interp_weights[:, v]
            if x.dim() == 1:
                out.index_add_(0, idx, w * x)
            else:
                out.index_add_(0, idx, w.unsqueeze(1) * x)
        return out

    def weights_u(self, u: torch.Tensor) -> torch.Tensor:
        """Compute W @ u via gathering."""
        # u is (actual_grid,) or (actual_grid, k)
        # result[i] = sum_j weight[i,j] * u[indices[i,j]]
        gathered = u[self.interp_indices]  # (n, vertices, [k])
        if u.dim() == 1:
            return (gathered * self.interp_weights).sum(dim=1)
        else:
            return (gathered * self.interp_weights.unsqueeze(-1)).sum(dim=1)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G_approx)`` to vector(s)."""
        out = self.lambda_reg * x
        v = self.weights_x(x)
        u = self.k_grid @ v
        out = out + self.weights_u(u)
        return out

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G_approx``."""
        # Diagonal approx = sum_j (W_{ij})^2 * K_grid[j,j]
        k_diag = self.k_grid.diagonal()
        gathered = k_diag[self.interp_indices]  # (n, vertices)
        diag_approx = (gathered * (self.interp_weights**2)).sum(dim=1)
        return self.lambda_reg + diag_approx

    def to_dense(self) -> torch.Tensor:
        """Materialise full approximate kernel matrix."""
        n = self.n
        g = self.grid_points.shape[0]
        w_dense = torch.zeros(n, g, device=self.device, dtype=self.dtype)
        vertices = self.interp_indices.shape[1]
        for v in range(vertices):
            w_dense[torch.arange(n), self.interp_indices[:, v]] += self.interp_weights[:, v]
        k_approx = w_dense @ self.k_grid @ w_dense.T
        k_approx.diagonal().add_(self.lambda_reg)
        return k_approx

    def build_interp_matrix(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return interpolation indices and weights for arbitrary points."""
        d = self.embedding_dim
        grid_points_per_dimension = self.grid_points_per_dim
        grid_1d_norm = [
            torch.linspace(
                0.0,
                1.0,
                grid_points_per_dimension,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(d)
        ]
        mins = torch.stack([g[0] for g in self.grid_1d])
        maxs = torch.stack([g[-1] for g in self.grid_1d])
        denom = maxs - mins
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        norm_pts = (points - mins) / denom
        return multilinear_weights(norm_pts, grid_1d_norm)

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate approximate kernel between queries and training points.

        For SKI this builds interpolation weights for ``x`` and returns
        ``W_x @ K_grid @ W_y^T``.
        """
        if y is None:
            y = self.embeddings
        p = y.shape[0]

        idx_x, w_x = self.build_interp_matrix(x)
        idx_y, w_y = self.build_interp_matrix(y)

        g = self.grid_points.shape[0]
        # Compute v = K_grid @ W_y^T  -- shape (g, p)
        v = torch.zeros(g, p, device=self.device, dtype=self.dtype)
        vertices = idx_y.shape[1]
        for vertex_index in range(vertices):
            idx = idx_y[:, vertex_index]  # (p,)
            w = w_y[:, vertex_index]  # (p,)
            # v[:, j] += K_grid[:, idx[j]] * w[j]
            v += self.k_grid[:, idx] * w.unsqueeze(0)

        # result = W_x @ v via gather
        gathered = v[idx_x]  # (m, vertices, p)
        out = (gathered * w_x.unsqueeze(-1)).sum(dim=1)  # (m, p)
        return out


class TwoScaleAttentionKernelOperator:
    """Two-scale kernel operator combining global low-rank + local sparse k-NN.

    Computes ``K = alpha * K_global + (1 - alpha) * K_local`` where:
    - ``K_global`` is a Nyström low-rank approximation.
    - ``K_local`` is a sparse k-NN graph in embedding space.

    This captures both global coherence and local sharpness, reducing
    oversmoothing compared to either approximation alone.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        alpha: Mixing coefficient in ``[0, 1]`` (default 0.5).
        num_landmarks: Number of landmarks for the global Nyström component.
        k_neighbors: Number of neighbours for the local sparse component.
        chunk_size: Chunk size for kernel evaluations.
        device: torch device.
        dtype: torch dtype.

    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        alpha: float = 0.5,
        num_landmarks: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        chunk_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise the spectrum shaper."""
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)
        self.alpha = float(alpha)
        self.chunk_size = chunk_size

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype

        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)
        self.shape = (self.n, self.n)

        self.global_op = NystromAttentionKernelOperator(
            embeddings=self.embeddings,
            lambda_reg=self.lambda_reg,
            num_landmarks=num_landmarks,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
        )
        self.local_op = SparseKNNAttentionKernelOperator(
            embeddings=self.embeddings,
            lambda_reg=self.lambda_reg,
            k_neighbors=k_neighbors,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
        )

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G_twoscale)`` to vector(s) ``x``."""
        return self.alpha * self.global_op.matvec(x) + (1.0 - self.alpha) * self.local_op.matvec(x)

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G_twoscale``."""
        return (
            self.alpha * self.global_op.diagonal() + (1.0 - self.alpha) * self.local_op.diagonal()
        )

    def to_dense(self) -> torch.Tensor:
        """Materialise full dense matrix (for debugging only)."""
        return (
            self.alpha * self.global_op.to_dense() + (1.0 - self.alpha) * self.local_op.to_dense()
        )

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate two-scale kernel between queries and training points."""
        return self.alpha * self.global_op.kernel_eval(x, y, chunk_size=chunk_size) + (
            1.0 - self.alpha
        ) * self.local_op.kernel_eval(x, y, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Spectral-shaped attention kernel
# ---------------------------------------------------------------------------


class MonotoneSpectrumShaper(nn.Module):
    """Learned monotone function applied to eigenvalues.

    Parameterised as a positive linear combination of shifted softplus
    functions plus a positive linear term.  Monotonicity is enforced by
    construction (all coefficients are positive), so the shaped spectrum
    preserves the positive-semidefinite property of the kernel.

    Args:
        num_knots: Number of fixed knot locations for the softplus basis.

    """

    def __init__(self, num_knots: int = 5) -> None:
        """Initialise the spectrum shaper."""
        super().__init__()
        self.num_knots = num_knots
        self.raw_weights = nn.Parameter(torch.full((num_knots,), -10.0))
        self.raw_slope = nn.Parameter(torch.tensor(-2.35))
        self.knots: Optional[torch.Tensor] = None

    def set_knots(
        self,
        min_val: float,
        max_val: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Place fixed knots linearly across the eigenvalue range."""
        self.knots = torch.linspace(min_val, max_val, self.num_knots, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned monotone function to ``x``."""
        if self.knots is None:
            raise RuntimeError("Knots not set. Call set_knots() first.")
        slope = torch.nn.functional.softplus(self.raw_slope)
        weights = torch.nn.functional.softplus(self.raw_weights)
        # x: (,) or (n,) -> out: same shape
        diffs = x.unsqueeze(-1) - self.knots  # (..., num_knots)
        out = slope * x + (weights * torch.nn.functional.softplus(diffs)).sum(dim=-1)
        return out


class SpectralAttentionKernelOperator:
    r"""Spectral-shaped attention kernel via matrix function of the embedding Gram matrix.

    Computes the kernel as a matrix function of ``S = E E^T``:

    .. math::
        K = U \, \operatorname{diag}\!\bigl(\exp(g(\sigma_i^2))\bigr) \, U^T

    where ``E = U \Sigma V^T`` is the (economy) SVD of the embedding matrix and
    ``g`` is a learned :class:`MonotoneSpectrumShaper`.  Because we SVD the
    ``n \times d`` embedding matrix directly, the operator costs only
    ``O(n d^2)`` to build and ``O(n d)`` per matvec — the same asymptotic
    cost as the standard attention kernel but with direct spectral control.

    For cross-evaluation (``kernel_eval``) the query embeddings are projected
    onto the training spectral basis, giving a consistent kernel matrix.

    Args:
        embeddings: Tensor of shape ``(n, embedding_dim)``.
        lambda_reg: Tikhonov regularisation ``lambda``.
        num_knots: Number of knots for the monotone spectrum shaper.
        device: torch device.
        dtype: torch dtype.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.

    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lambda_reg: float = 1e-2,
        num_knots: int = 5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise the spectral-shaped kernel operator."""
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.n = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.lambda_reg = float(lambda_reg)

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype
        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)
        self.shape = (self.n, self.n)

        # Economy SVD of the embedding matrix (n x d)
        u, s, vh = torch.linalg.svd(self.embeddings, full_matrices=False)
        self.u_matrix = u  # (n, d)
        self.sigma = s  # (d,)
        self.vh = vh  # (d, d)
        sigma_sq = s**2  # eigenvalues of S = E E^T

        # Learned monotone shaper
        self.shaper = MonotoneSpectrumShaper(num_knots=num_knots)
        min_val = float(sigma_sq.min().item())
        max_val = float(sigma_sq.max().item())
        # Add a little padding so knots cover the range comfortably
        pad = max(1e-6, (max_val - min_val) * 0.1)
        self.shaper.set_knots(min_val - pad, max_val + pad, device=device, dtype=dtype)
        self.shaper = self.shaper.to(device=device, dtype=dtype)

        with torch.no_grad():
            shaped = self.shaper(sigma_sq)
            max_exp = 80.0 if dtype == torch.float32 else 700.0
            self.spectrum = torch.exp(shaped.clamp(max=max_exp))  # (d,)

        self.sigma_inv = torch.where(s > 1e-12, s.reciprocal(), torch.zeros_like(s))

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + K)`` to vector(s) ``x``."""
        out = self.lambda_reg * x
        # U^T @ x -> (d,) or (d, k)
        coeffs = self.u_matrix.T @ x
        if coeffs.dim() == 1:
            scaled = self.spectrum * coeffs
        else:
            scaled = self.spectrum.unsqueeze(-1) * coeffs
        out = out + self.u_matrix @ scaled
        return out

    def diagonal(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + K``."""
        # diag(K)_j = sum_i spectrum_i * U_{j,i}^2
        diag_k = (self.u_matrix**2) @ self.spectrum
        return self.lambda_reg + diag_k

    def to_dense(self) -> torch.Tensor:
        """Materialise full dense matrix (for debugging only)."""
        k_dense = self.u_matrix @ torch.diag(self.spectrum) @ self.u_matrix.T
        k_dense.diagonal().add_(self.lambda_reg)
        return k_dense

    def kernel_eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Evaluate shaped kernel between queries and reference points.

        Both ``x`` and ``y`` are projected onto the training spectral basis
        (derived from the SVD of the training embeddings), so the result is
        consistent with the training-training kernel ``K = U diag(spectrum) U^T``.
        """
        if y is None:
            y = self.embeddings
        # C_x = x @ V @ Sigma^{-1}   (m, d)
        cx = (x @ self.vh.T) * self.sigma_inv.unsqueeze(0)
        # C_y = y @ V @ Sigma^{-1}   (n_ref, d)
        cy = (y @ self.vh.T) * self.sigma_inv.unsqueeze(0)
        # K = C_x @ diag(spectrum) @ C_y^T
        k = (cx * self.spectrum.unsqueeze(0)) @ cy.T
        return k
