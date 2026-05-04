"""Core LAKER pipeline: embeddings, kernels, preconditioners, solvers, predictions."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union, cast

import torch
import torch.nn as nn

from laker.backend import get_default_device, get_default_dtype
from laker.distributed_kernels import DistributedAttentionKernelOperator
from laker.embeddings import PositionEmbedding
from laker.kernels import (
    AttentionKernelOperator,
    KernelOperator,
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
    SKIAttentionKernelOperator,
    SparseKNNAttentionKernelOperator,
    SpectralAttentionKernelOperator,
    TwoScaleAttentionKernelOperator,
    exp_safe,
)
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient

logger = logging.getLogger(__name__)


class LAKERCore:
    """Encapsulates the standard LAKER solve/predict pipeline.

    Stores hyperparameters and provides methods that operate on fitted
    state passed as arguments.  Designed to be composed by
    ``LAKERRegressor``.
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        lambda_reg: float = 1e-2,
        gamma: float = 1e-1,
        num_probes: Optional[int] = None,
        epsilon: float = 1e-8,
        base_rho: float = 0.05,
        cccp_max_iter: int = 200,
        cccp_tol: float = 1e-6,
        pcg_tol: float = 1e-6,
        pcg_max_iter: int = 1000,
        chunk_size: Optional[int] = None,
        embedding_module: Optional[nn.Module] = None,
        kernel_approx: Optional[str] = None,
        num_landmarks: Optional[int] = None,
        num_features: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        grid_size: Optional[int] = None,
        distributed: bool = False,
        twoscale_alpha: float = 0.5,
        landmark_method: str = "greedy",
        landmark_pilot_size: int = 1000,
        spectral_knots: int = 5,
        preconditioner_strategy: str = "cccp",
        embedding_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = True,
    ) -> None:
        """Initialise the core LAKER pipeline."""
        self.embedding_dim = embedding_dim
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.num_probes = num_probes
        self.epsilon = epsilon
        self.base_rho = base_rho
        self.cccp_max_iter = cccp_max_iter
        self.cccp_tol = cccp_tol
        self.pcg_tol = pcg_tol
        self.pcg_max_iter = pcg_max_iter
        self.chunk_size = chunk_size
        self.embedding_module = embedding_module
        self.kernel_approx = kernel_approx
        self.num_landmarks = num_landmarks
        self.num_features = num_features
        self.k_neighbors = k_neighbors
        self.grid_size = grid_size
        self.distributed = distributed
        self.twoscale_alpha = twoscale_alpha
        self.landmark_method = landmark_method
        self.landmark_pilot_size = landmark_pilot_size
        self.spectral_knots = spectral_knots
        self.preconditioner_strategy = preconditioner_strategy
        self.verbose = verbose

        if device is None:
            device = get_default_device()
        elif isinstance(device, str):
            device = torch.device(device)
        if dtype is None:
            dtype = get_default_dtype()
        self.device = device
        self.dtype = dtype
        self.embedding_dtype = embedding_dtype if embedding_dtype is not None else dtype

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def compute_embeddings(
        self, x: torch.Tensor, embedding_model: Optional[nn.Module] = None
    ) -> tuple[torch.Tensor, nn.Module]:
        """Build or reuse embeddings from spatial locations.

        Returns:
            Tuple of ``(embeddings, embedding_model)``.

        """
        n = x.shape[0]
        input_dim = x.shape[1]

        if self.verbose:
            logger.info("Fitting LAKER on n=%d, dx=%d", n, input_dim)

        embedded_input = x.to(dtype=self.embedding_dtype)
        if self.embedding_module is not None:
            model = self.embedding_module.to(self.device)
            with torch.no_grad():
                embeddings = model(embedded_input)
        else:
            if embedding_model is None:
                model = PositionEmbedding(
                    input_dim=input_dim,
                    embedding_dim=self.embedding_dim,
                    device=self.device,
                    dtype=self.embedding_dtype,
                )
            else:
                model = embedding_model
            with torch.no_grad():
                embeddings = model(embedded_input)

        if self.embedding_dtype != self.dtype:
            embeddings = embeddings.to(dtype=self.dtype)
            if self.verbose:
                logger.info(
                    "Mixed-precision: embeddings computed in %s, cast to %s for solver",
                    self.embedding_dtype,
                    self.dtype,
                )
        return embeddings, model

    # ------------------------------------------------------------------
    # Kernel operator
    # ------------------------------------------------------------------
    def build_kernel_operator(
        self,
        embeddings: torch.Tensor,
        lambda_reg: Optional[float] = None,
        chunk_size: Optional[int] = None,
    ) -> KernelOperator:
        """Construct the kernel operator for given embeddings."""
        n = embeddings.shape[0]
        lambda_value = float(lambda_reg) if lambda_reg is not None else self.lambda_reg
        chunk_size_local = chunk_size
        if chunk_size_local is None and n > 5000:
            chunk_size_local = max(1024, min(n // 10, 8192))
            if self.verbose:
                logger.info("Auto-selected chunk_size=%d for n=%d", chunk_size_local, n)

        operator: KernelOperator
        if self.distributed and self.kernel_approx is None:
            operator = DistributedAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                master_device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using distributed kernel on %d device(s)",
                    len(cast(DistributedAttentionKernelOperator, operator).devices),
                )
        elif self.kernel_approx is None:
            operator = AttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                chunk_size=chunk_size_local,
                device=self.device,
                dtype=self.dtype,
            )
        elif self.kernel_approx == "nystrom":
            operator = NystromAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                num_landmarks=self.num_landmarks,
                landmark_method=self.landmark_method,
                landmark_pilot_size=self.landmark_pilot_size,
                chunk_size=chunk_size_local,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using Nyström approximation with m=%d landmarks (%s)",
                    cast(NystromAttentionKernelOperator, operator).m,
                    self.landmark_method,
                )
        elif self.kernel_approx == "rff":
            operator = RandomFeatureAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                num_features=self.num_features,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using RFF approximation with r=%d features",
                    cast(RandomFeatureAttentionKernelOperator, operator).num_features,
                )
        elif self.kernel_approx == "knn":
            operator = SparseKNNAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                k_neighbors=self.k_neighbors,
                chunk_size=chunk_size_local,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using sparse k-NN approximation with k=%d neighbours",
                    cast(SparseKNNAttentionKernelOperator, operator).k_neighbors,
                )
        elif self.kernel_approx == "ski":
            operator = SKIAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                grid_size=self.grid_size,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using SKI approximation with %d grid points",
                    cast(SKIAttentionKernelOperator, operator).grid_points.shape[0],
                )
        elif self.kernel_approx == "twoscale":
            operator = TwoScaleAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                alpha=getattr(self, "twoscale_alpha", 0.5),
                num_landmarks=self.num_landmarks,
                k_neighbors=self.k_neighbors,
                chunk_size=chunk_size_local,
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using two-scale kernel (alpha=%.2f)",
                    cast(TwoScaleAttentionKernelOperator, operator).alpha,
                )
        elif self.kernel_approx == "spectral":
            operator = SpectralAttentionKernelOperator(
                embeddings=embeddings,
                lambda_reg=lambda_value,
                num_knots=getattr(self, "spectral_knots", 5),
                device=self.device,
                dtype=self.dtype,
            )
            if self.verbose:
                logger.info(
                    "Using spectral-shaped kernel with %d knots",
                    cast(SpectralAttentionKernelOperator, operator).shaper.num_knots,
                )
        else:
            raise ValueError(f"Unknown kernel_approx={self.kernel_approx}")
        return operator

    # ------------------------------------------------------------------
    # Preconditioner
    # ------------------------------------------------------------------
    def build_preconditioner(
        self,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        n: int,
        gamma: Optional[float] = None,
        num_probes: Optional[int] = None,
        seed: Optional[int] = None,
        diagonal: Optional[torch.Tensor] = None,
    ) -> CCCPPreconditioner:
        """Learn the preconditioner for a given matvec."""
        if self.preconditioner_strategy == "adaptive":
            from laker.preconditioner import AdaptivePreconditioner

            preconditioner = AdaptivePreconditioner(
                gamma=gamma if gamma is not None else self.gamma,
                num_probes=(num_probes if num_probes is not None else self.num_probes),
                epsilon=self.epsilon,
                base_rho=self.base_rho,
                max_iter=self.cccp_max_iter,
                tol=self.cccp_tol,
                verbose=self.verbose,
                device=self.device,
                dtype=self.dtype,
            )
            preconditioner.build(matvec, n, diagonal=diagonal, seed=seed)
            return preconditioner

        preconditioner = CCCPPreconditioner(
            num_probes=(num_probes if num_probes is not None else self.num_probes),
            gamma=gamma if gamma is not None else self.gamma,
            epsilon=self.epsilon,
            base_rho=self.base_rho,
            max_iter=self.cccp_max_iter,
            tol=self.cccp_tol,
            verbose=self.verbose,
            device=self.device,
            dtype=self.dtype,
        )
        preconditioner.build(matvec, n, seed=seed)
        return preconditioner

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    def solve_pcg(
        self,
        kernel_operator: KernelOperator,
        preconditioner: CCCPPreconditioner,
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        """Solve (K + lambda I) alpha = rhs with PCG."""
        pcg = PreconditionedConjugateGradient(
            tol=self.pcg_tol,
            max_iter=self.pcg_max_iter,
            verbose=self.verbose,
        )
        alpha = pcg.solve(
            operator=kernel_operator.matvec,
            preconditioner=preconditioner.apply,
            rhs=rhs,
            x0=x0,
        )
        if self.verbose:
            final_res = (
                torch.linalg.norm(kernel_operator.matvec(alpha) - rhs).item()
                / torch.linalg.norm(rhs).item()
            )
            logger.info(
                "LAKER fit complete: PCG iters=%d, final rel_res=%.3e",
                pcg.iterations,
                final_res,
            )
        return alpha, pcg.iterations

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        x: torch.Tensor,
        embedding_model: nn.Module,
        embeddings: torch.Tensor,
        kernel_operator: KernelOperator,
        alpha: torch.Tensor,
        residual_corrector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Reconstruct the radio field at query locations."""
        with torch.no_grad():
            embedded_input = x.to(dtype=self.embedding_dtype)
            query_embeddings = embedding_model(embedded_input)
            if self.embedding_dtype != self.dtype:
                query_embeddings = query_embeddings.to(dtype=self.dtype)
            m = query_embeddings.shape[0]
            n = embeddings.shape[0]

            chunk_size = self.chunk_size
            if chunk_size is None and max(m, n) > 5000:
                chunk_size = max(1024, min(max(m, n) // 10, 8192))

            element_size = 4 if self.dtype == torch.float32 else 8
            mem_per_chunk = (
                (chunk_size or m) * n * element_size if chunk_size else m * n * element_size
            )
            if chunk_size is None or mem_per_chunk <= 64 * 1024 * 1024:
                k_query = kernel_operator.kernel_eval(
                    query_embeddings, embeddings, chunk_size=chunk_size
                )
                out = k_query @ alpha
            elif self.kernel_approx is not None:
                k_query = kernel_operator.kernel_eval(
                    query_embeddings, embeddings, chunk_size=chunk_size
                )
                out = k_query @ alpha
            else:
                out = torch.empty(m, device=self.device, dtype=self.dtype)
                chunk_size_local = chunk_size
                for i_start in range(0, m, chunk_size_local):
                    i_end = min(i_start + chunk_size_local, m)
                    accum = torch.zeros(i_end - i_start, device=self.device, dtype=self.dtype)
                    e_i = query_embeddings[i_start:i_end]
                    for j_start in range(0, n, chunk_size_local):
                        j_end = min(j_start + chunk_size_local, n)
                        gram_block = e_i @ embeddings[j_start:j_end].T
                        exp_safe(gram_block, out=gram_block)
                        accum.addmv_(gram_block, alpha[j_start:j_end])
                    out[i_start:i_end] = accum

            if residual_corrector is not None:
                residual_corrector.eval()
                with torch.no_grad():
                    out = out + residual_corrector(x).squeeze()
            return out

    # ------------------------------------------------------------------
    # Variance
    # ------------------------------------------------------------------
    def predict_variance(
        self,
        x: torch.Tensor,
        embedding_model: nn.Module,
        embeddings: torch.Tensor,
        kernel_operator: KernelOperator,
        preconditioner: CCCPPreconditioner,
        alpha: torch.Tensor,
        lambda_reg: float,
    ) -> torch.Tensor:
        """Predictive variance (uncertainty) at query locations."""
        with torch.no_grad():
            embedded_input = x.to(dtype=self.embedding_dtype)
            query_embeddings = embedding_model(embedded_input)
            if self.embedding_dtype != self.dtype:
                query_embeddings = query_embeddings.to(dtype=self.dtype)
            m = query_embeddings.shape[0]
            n = embeddings.shape[0]

            if self.kernel_approx == "rff" and hasattr(kernel_operator, "phi"):
                ko = cast(RandomFeatureAttentionKernelOperator, kernel_operator)
                proj = query_embeddings @ ko.freq
                phi_q = torch.cat(
                    [torch.cos(proj + ko.phase), torch.sin(proj + ko.phase)],
                    dim=1,
                ) / (ko.num_features**0.5)
                a = ko.phi.T @ ko.phi
                a_reg = a + lambda_reg * torch.eye(a.shape[0], device=self.device, dtype=self.dtype)
                chol = torch.linalg.cholesky(a_reg)
                m_solve = torch.cholesky_solve(
                    torch.eye(a.shape[0], device=self.device, dtype=self.dtype),
                    chol,
                )
                var = lambda_reg * torch.sum(phi_q @ m_solve * phi_q, dim=1)
                return var.clamp(min=0.0)

            element_size = 4 if self.dtype == torch.float32 else 8
            chunk_size = self.chunk_size
            if chunk_size is None:
                mem_needed = m * n * element_size
                if mem_needed > 64 * 1024 * 1024:
                    chunk_size = max(1024, min(n // 10, 8192))

            var = torch.empty(m, device=self.device, dtype=self.dtype)
            pcg = PreconditionedConjugateGradient(
                tol=self.pcg_tol,
                max_iter=self.pcg_max_iter,
                verbose=False,
            )

            if chunk_size is None or m <= chunk_size:
                k_train_query = kernel_operator.kernel_eval(embeddings, query_embeddings)
                if k_train_query.is_sparse:
                    k_train_query = k_train_query.to_dense()
                v = pcg.solve(
                    operator=kernel_operator.matvec,
                    preconditioner=preconditioner.apply,
                    rhs=k_train_query,
                )
                k_diag_mat = kernel_operator.kernel_eval(query_embeddings, query_embeddings)
                if k_diag_mat.is_sparse:
                    k_diag_mat = k_diag_mat.to_dense()
                k_diag = k_diag_mat.diagonal()
                var[:] = k_diag - torch.sum(k_train_query * v, dim=0)
            else:
                for start in range(0, m, chunk_size):
                    end = min(start + chunk_size, m)
                    q_chunk = query_embeddings[start:end]
                    k_train_chunk = kernel_operator.kernel_eval(embeddings, q_chunk)
                    if k_train_chunk.is_sparse:
                        k_train_chunk = k_train_chunk.to_dense()
                    v_chunk = pcg.solve(
                        operator=kernel_operator.matvec,
                        preconditioner=preconditioner.apply,
                        rhs=k_train_chunk,
                    )
                    k_diag_mat = kernel_operator.kernel_eval(q_chunk, q_chunk)
                    if k_diag_mat.is_sparse:
                        k_diag_mat = k_diag_mat.to_dense()
                    k_diag_chunk = k_diag_mat.diagonal()
                    var[start:end] = k_diag_chunk - torch.sum(k_train_chunk * v_chunk, dim=0)

            return var.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Differentiable prediction / variance (for training loops)
    # ------------------------------------------------------------------
    def predict_train(
        self,
        x: torch.Tensor,
        embedding_model: nn.Module,
        embeddings: torch.Tensor,
        kernel_operator: KernelOperator,
        alpha: torch.Tensor,
        residual_corrector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Differentiable version of :meth:`predict` (no ``torch.no_grad``)."""
        embedded_input = x.to(dtype=self.embedding_dtype)
        query_embeddings = embedding_model(embedded_input)
        if self.embedding_dtype != self.dtype:
            query_embeddings = query_embeddings.to(dtype=self.dtype)
        m = query_embeddings.shape[0]
        n = embeddings.shape[0]

        chunk_size = self.chunk_size
        if chunk_size is None and max(m, n) > 5000:
            chunk_size = max(1024, min(max(m, n) // 10, 8192))

        element_size = 4 if self.dtype == torch.float32 else 8
        mem_per_chunk = (chunk_size or m) * n * element_size if chunk_size else m * n * element_size
        if chunk_size is None or mem_per_chunk <= 64 * 1024 * 1024:
            k_query = kernel_operator.kernel_eval(
                query_embeddings, embeddings, chunk_size=chunk_size
            )
            out = k_query @ alpha
        elif self.kernel_approx is not None:
            k_query = kernel_operator.kernel_eval(
                query_embeddings, embeddings, chunk_size=chunk_size
            )
            out = k_query @ alpha
        else:
            out = torch.empty(m, device=self.device, dtype=self.dtype)
            chunk_size_local = chunk_size
            for i_start in range(0, m, chunk_size_local):
                i_end = min(i_start + chunk_size_local, m)
                accum = torch.zeros(i_end - i_start, device=self.device, dtype=self.dtype)
                e_i = query_embeddings[i_start:i_end]
                for j_start in range(0, n, chunk_size_local):
                    j_end = min(j_start + chunk_size_local, n)
                    gram_block = e_i @ embeddings[j_start:j_end].T
                    exp_safe(gram_block, out=gram_block)
                    accum.addmv_(gram_block, alpha[j_start:j_end])
                out[i_start:i_end] = accum

        if residual_corrector is not None:
            residual_corrector.eval()
            out = out + residual_corrector(x).squeeze()
        return out

    def predict_variance_train(
        self,
        x: torch.Tensor,
        embedding_model: nn.Module,
        embeddings: torch.Tensor,
        kernel_operator: KernelOperator,
        preconditioner: CCCPPreconditioner,
        alpha: torch.Tensor,
        lambda_reg: float,
    ) -> torch.Tensor:
        """Differentiable predictive variance for training loops.

        For the RFF kernel the exact closed-form variance is used (fully
        differentiable).  For other kernels a differentiable proxy based on
        the distance to the nearest training embedding in the spectral basis is
        returned.  This proxy increases when a query lies far from the training
        manifold and is stable for back-propagation.
        """
        embedded_input = x.to(dtype=self.embedding_dtype)
        query_embeddings = embedding_model(embedded_input)
        if self.embedding_dtype != self.dtype:
            query_embeddings = query_embeddings.to(dtype=self.dtype)

        if self.kernel_approx == "rff" and hasattr(kernel_operator, "phi"):
            ko = cast(RandomFeatureAttentionKernelOperator, kernel_operator)
            proj = query_embeddings @ ko.freq
            phi_q = torch.cat([torch.cos(proj + ko.phase), torch.sin(proj + ko.phase)], dim=1) / (
                ko.num_features**0.5
            )
            a = ko.phi.T @ ko.phi
            a_reg = a + lambda_reg * torch.eye(a.shape[0], device=self.device, dtype=self.dtype)
            chol = torch.linalg.cholesky(a_reg)
            m_solve = torch.cholesky_solve(
                torch.eye(a.shape[0], device=self.device, dtype=self.dtype),
                chol,
            )
            var = lambda_reg * torch.sum(phi_q @ m_solve * phi_q, dim=1)
            return var.clamp(min=0.0)

        # Differentiable proxy: soft-min distance to training embeddings
        dists = torch.cdist(query_embeddings, embeddings) ** 2
        weights = torch.softmax(-dists, dim=1)
        mean_dist = (weights * dists).sum(dim=1)
        return (lambda_reg + mean_dist).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def condition_number(
        self,
        kernel_operator: KernelOperator,
        preconditioner: CCCPPreconditioner,
    ) -> float:
        """Return the condition number of the preconditioned system."""
        n = kernel_operator.n
        apply_operator = kernel_operator.matvec
        apply_precond = preconditioner.apply

        def preconditioned_operator(v: torch.Tensor) -> torch.Tensor:
            return apply_precond(apply_operator(v))

        v = torch.randn(n, device=self.device, dtype=self.dtype)
        v = v / torch.linalg.norm(v)
        for _ in range(10):
            v = preconditioned_operator(v)
            v = v / torch.linalg.norm(v)
        lam_max = torch.dot(v, preconditioned_operator(v)).item()

        v = torch.randn(n, device=self.device, dtype=self.dtype)
        v = v / torch.linalg.norm(v)
        pcg = PreconditionedConjugateGradient(tol=1e-6, max_iter=50, verbose=False)
        for _ in range(5):
            v = pcg.solve(
                operator=preconditioned_operator,
                preconditioner=lambda x: x,
                rhs=v,
            )
            v = v / torch.linalg.norm(v)
        lam_min = max(
            torch.dot(v, preconditioned_operator(v)).item(),
            torch.finfo(self.dtype).eps,
        )

        return lam_max / lam_min
