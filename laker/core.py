"""Core LAKER pipeline: embeddings, kernels, preconditioners, solvers, predictions.

This module defines :class:`LAKERCore`, the central composition root that
wires together every stage of the LAKER regression pipeline:

1. **Embeddings** -- Maps raw spatial coordinates
   :math:`x \\in \\mathbb{R}^d` to a learned feature space
   :math:`E(x) \\in \\mathbb{R}^{D}` via either a fixed positional
   encoding or a trainable neural network
   (:meth:`LAKERCore.compute_embeddings`).

2. **Kernel operator** -- Constructs the attention kernel
   :math:`K = \\exp(E E^\\top) + \\lambda I` from the embeddings.  Supports
   the exact kernel, Nyström approximation, random Fourier features (RFF),
   sparse k-NN, SKI (structured kernel interpolation), two-scale, and
   spectral-shaped variants
   (:meth:`LAKERCore.build_kernel_operator`).

3. **Preconditioner** -- Learns a CCCP (concave-convex composite programming)
   or adaptive preconditioner for the kernel matrix to accelerate the PCG
   solver (:meth:`LAKERCore.build_preconditioner`).

4. **PCG solver** -- Solves the regularised linear system
   :math:`(K + \\lambda I)\\alpha = y` using preconditioned conjugate
   gradients (:meth:`LAKERCore.solve_pcg`).

5. **Prediction** -- Reconstructs the field at arbitrary query locations by
   evaluating the kernel between query embeddings and training embeddings,
   optionally adding a residual corrector output
   (:meth:`LAKERCore.predict`, :meth:`LAKERCore.predict_variance`).

6. **Training variants** -- Differentiable versions of predict and
   variance (:meth:`LAKERCore.predict_train`,
   :meth:`LAKERCore.predict_variance_train`) that omit
   ``torch.no_grad`` blocks so gradients can flow through the pipeline
   during end-to-end embedding optimisation.

7. **Diagnostics** -- Condition-number estimation for the preconditioned
   system (:meth:`LAKERCore.condition_number`).

``LAKERCore`` is stateless with respect to fitted data; it stores only
hyperparameters and device/dtype configuration.  Fitted state (embeddings,
alpha, kernel operator, preconditioner) lives on the
:class:`~laker.models.LAKERRegressor` and is passed as arguments to each
method.  This separation allows ``LAKERCore`` to be composed and reused
by the high-level estimator, the streaming updater, and training loops.
"""

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
from laker.preconditioner import AdaptivePreconditioner, CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient

logger = logging.getLogger(__name__)


class LAKERCore:
    """Encapsulates the standard LAKER solve/predict pipeline.

    Stores hyperparameters and provides methods that operate on fitted
    state passed as arguments.  Designed to be composed by
    ``LAKERRegressor``.

    The pipeline stages are:

    * :meth:`compute_embeddings` -- spatial coordinates to embedding vectors.
    * :meth:`build_kernel_operator` -- embeddings to regularised attention
      kernel.
    * :meth:`build_preconditioner` -- kernel matvec to CCCP/adaptive
      preconditioner.
    * :meth:`solve_pcg` -- preconditioned conjugate gradient solve for
      :math:`\\alpha`.
    * :meth:`predict` / :meth:`predict_variance` -- interpolation and
      uncertainty at query points.

    Args:
        embedding_dim: Dimensionality of the embedding space :math:`D`.
        lambda_reg: Regularisation weight :math:`\\lambda` in the kernel
            ridge regression objective.
        gamma: Kernel bandwidth parameter for the CCCP preconditioner.
        num_probes: Number of probe vectors for the CCCP preconditioner.
            If ``None``, a default is selected automatically.
        epsilon: Small constant for numerical stability in the
            preconditioner.
        base_rho: Base spectral norm bound for the CCCP preconditioner.
        cccp_max_iter: Maximum CCCP iterations when building the
            preconditioner.
        cccp_tol: Convergence tolerance for the CCCP preconditioner.
        pcg_tol: Relative residual tolerance for the PCG solver.
        pcg_max_iter: Maximum number of PCG iterations.
        chunk_size: Tile size for block-sparse kernel evaluations.  If
            ``None``, chosen automatically based on dataset size.
        embedding_module: Optional pre-built ``nn.Module`` for computing
            embeddings.  If ``None``, a positional encoding is used.
        kernel_approx: Kernel approximation method.  One of ``None`` (exact),
            ``"nystrom"``, ``"rff"``, ``"knn"``, ``"ski"``, ``"twoscale"``,
            or ``"spectral"``.
        num_landmarks: Number of landmark points for Nyström or two-scale
            approximations.
        num_features: Number of random Fourier features for RFF
            approximation.
        k_neighbors: Number of nearest neighbours for sparse k-NN kernel.
        grid_size: Grid resolution for SKI approximation.
        distributed: If ``True``, use distributed kernel evaluation across
            multiple devices.
        twoscale_alpha: Blending coefficient for the two-scale kernel.
        landmark_method: Landmark selection strategy (``"greedy"`` or
            ``"leverage"``).
        landmark_pilot_size: Pilot sample size for leverage-score landmark
            selection.
        spectral_knots: Number of spline knots for the spectral-shaped
            kernel.
        preconditioner_strategy: Preconditioner type (``"cccp"`` or
            ``"adaptive"``).
        embedding_dtype: Floating-point dtype for embedding computation.
            Defaults to ``dtype``.
        device: PyTorch device for computation.
        dtype: Floating-point dtype for kernel and solver computation.
        verbose: If ``True``, log diagnostic information during the pipeline.
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
        """Initialise the core LAKER pipeline.

        All parameters have sensible defaults.  The ``device`` and ``dtype``
        are resolved via :mod:`laker.backend` if not provided.

        Args:
            embedding_dim: Dimensionality of the learned embedding space.
            lambda_reg: Regularisation weight in the kernel ridge regression
                objective.
            gamma: Kernel bandwidth for the CCCP preconditioner.
            num_probes: Number of random probe vectors for preconditioner
                construction.
            epsilon: Numerical stability constant for the preconditioner.
            base_rho: Base spectral norm bound for the CCCP preconditioner.
            cccp_max_iter: Maximum CCCP iterations.
            cccp_tol: CCCP convergence tolerance.
            pcg_tol: PCG relative residual tolerance.
            pcg_max_iter: Maximum PCG iterations.
            chunk_size: Tile size for chunked kernel evaluations.
            embedding_module: Optional pre-built embedding ``nn.Module``.
            kernel_approx: Kernel approximation method string.
            num_landmarks: Landmarks for Nyström / two-scale kernels.
            num_features: Random Fourier features for RFF kernel.
            k_neighbors: k-NN sparsity for sparse kernel.
            grid_size: Grid resolution for SKI kernel.
            distributed: Whether to use multi-device distributed kernel.
            twoscale_alpha: Blending weight for two-scale kernel.
            landmark_method: Landmark selection strategy.
            landmark_pilot_size: Pilot size for leverage-score landmarks.
            spectral_knots: Number of spline knots for spectral kernel.
            preconditioner_strategy: ``"cccp"`` or ``"adaptive"``.
            embedding_dtype: Dtype for embedding computation (defaults to
                ``dtype``).
            device: Target PyTorch device.
            dtype: Target floating-point dtype.
            verbose: Whether to emit diagnostic log messages.
        """
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

        If ``self.embedding_module`` is set, it is used directly.
        Otherwise, if an ``embedding_model`` is provided it is used; if
        neither is provided, a new
        :class:`~laker.embeddings.PositionEmbedding` is created.

        When ``embedding_dtype`` differs from ``dtype``, the output
        embeddings are cast to the solver dtype after computation.

        Args:
            x: Spatial locations of shape ``(n, d)``.
            embedding_model: Optional pre-built embedding module.  Ignored
                if ``self.embedding_module`` is set.

        Returns:
            A tuple ``(embeddings, embedding_model)`` where ``embeddings``
            has shape ``(n, D)`` and ``embedding_model`` is the module used
            to produce them.
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
        """Construct the kernel operator for given embeddings.

        Selects the appropriate :class:`~laker.kernels.KernelOperator`
        subclass based on ``self.kernel_approx`` and wraps the embeddings
        into an operator that supports ``matvec``, ``kernel_eval``, and
        ``diagonal`` methods.

        When the dataset exceeds 5000 points and no ``chunk_size`` is
        provided, a tile size is chosen automatically to bound memory
        usage.

        Args:
            embeddings: Training embeddings of shape ``(n, D)``.
            lambda_reg: Regularisation weight.  Defaults to
                ``self.lambda_reg``.
            chunk_size: Tile size for chunked kernel evaluations.  If
                ``None``, chosen automatically for large datasets.

        Returns:
            A :class:`~laker.kernels.KernelOperator` instance wrapping the
            attention kernel :math:`\\exp(E E^\\top) + \\lambda I`.

        Raises:
            ValueError: If ``self.kernel_approx`` is not a recognised
                string.
        """
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
    ) -> Union[CCCPPreconditioner, AdaptivePreconditioner]:
        """Learn the preconditioner for a given matvec.

        Builds either a CCCP or adaptive preconditioner depending on
        ``self.preconditioner_strategy``.  The preconditioner is learned
        by applying the kernel matvec to random probe vectors and
        fitting a spectral bound.

        Args:
            matvec: A callable ``matvec(v) -> Kv`` that computes the
                matrix-vector product with the regularised kernel matrix
                :math:`K = G + \\lambda I`.
            n: Number of training points (dimension of the system).
            gamma: Kernel bandwidth for the CCCP preconditioner.  Defaults
                to ``self.gamma``.
            num_probes: Number of random probe vectors.  Defaults to
                ``self.num_probes``.
            seed: Random seed for reproducibility of probe vectors.
            diagonal: Optional pre-computed diagonal of the kernel matrix.

        Returns:
            A :class:`~laker.preconditioner.CCCPPreconditioner` or
            :class:`~laker.preconditioner.AdaptivePreconditioner` instance
            ready to be passed to :meth:`solve_pcg`.
        """
        if self.preconditioner_strategy == "adaptive":
            adaptive_precond = AdaptivePreconditioner(
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
            adaptive_precond.build(matvec, n, diagonal=diagonal, seed=seed)
            return adaptive_precond

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
        preconditioner: Union[CCCPPreconditioner, AdaptivePreconditioner],
        rhs: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        """Solve (K + lambda I) alpha = rhs with PCG.

        Runs preconditioned conjugate gradients until the relative residual
        falls below ``pcg_tol`` or the iteration count reaches
        ``pcg_max_iter``.

        Args:
            kernel_operator: The kernel operator providing the ``matvec``
                method.
            preconditioner: A learned preconditioner providing an ``apply``
                method.
            rhs: Right-hand side vector of shape ``(n,)`` (typically the
                observation vector ``y``).
            x0: Optional initial guess for ``alpha``.  If ``None``, the
                zero vector is used.

        Returns:
            A tuple ``(alpha, iterations)`` where ``alpha`` is the solution
            vector of shape ``(n,)`` and ``iterations`` is the number of
            PCG iterations performed.
        """
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
        """Reconstruct the radio field at query locations.

        Evaluates the kernel interpolation

        .. math::

            \\hat{f}(x) = k(x, X)^\\top \\alpha

        where :math:`k(x, X)` is the attention kernel between the query
        embedding :math:`E(x)` and each training embedding.  An optional
        residual corrector output is added to the base prediction.

        For large datasets the computation is chunked to bound memory usage
        to approximately 64 MB per tile.

        Args:
            x: Query locations of shape ``(m, d)``.
            embedding_model: The embedding network used to map ``x`` to
                feature vectors.
            embeddings: Training embeddings of shape ``(n, D)``.
            kernel_operator: The fitted kernel operator.
            alpha: Solution vector of shape ``(n,)`` from :meth:`solve_pcg`.
            residual_corrector: Optional residual corrector ``nn.Module``.
                If provided, its output is added to the kernel prediction.

        Returns:
            Predicted field values of shape ``(m,)``.
        """
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
        """Predictive variance (uncertainty) at query locations.

        Computes the posterior variance

        .. math::

            \\sigma^2(x) = k(x, x) - k(x, X)^\\top (K + \\lambda I)^{-1} k(x, X)

        where :math:`K` is the training kernel matrix and
        :math:`k(x, X)` is the cross-kernel vector between the query and
        all training points.  The inverse is approximated via PCG.

        For the RFF kernel, an exact closed-form expression using the
        random feature basis is used, which avoids the PCG solve entirely.

        The result is clamped to be non-negative.

        Args:
            x: Query locations of shape ``(m, d)``.
            embedding_model: The embedding network.
            embeddings: Training embeddings of shape ``(n, D)``.
            kernel_operator: The fitted kernel operator.
            preconditioner: The learned preconditioner for PCG.
            alpha: Solution vector of shape ``(n,)``.
            lambda_reg: Regularisation weight :math:`\\lambda`.

        Returns:
            Predictive variance of shape ``(m,)``, clamped to
            :math:`\\geq 0`.
        """
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
        """Differentiable version of :meth:`predict` (no ``torch.no_grad``).

        Identical to :meth:`predict` but retains the computation graph so
        that gradients can flow through the kernel evaluation and the
        embedding model during training loops (e.g.
        :meth:`EmbeddingTrainer.fit_learned_embeddings`).

        Args:
            x: Query locations of shape ``(m, d)``.
            embedding_model: The embedding network (gradients flow through).
            embeddings: Training embeddings of shape ``(n, D)``.
            kernel_operator: The kernel operator.
            alpha: Solution vector of shape ``(n,)``.
            residual_corrector: Optional residual corrector module.

        Returns:
            Differentiable predicted field values of shape ``(m,)``.
        """
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
        preconditioner: Union[CCCPPreconditioner, AdaptivePreconditioner],
        alpha: torch.Tensor,
        lambda_reg: float,
    ) -> torch.Tensor:
        """Differentiable predictive variance for training loops.

        For the RFF kernel the exact closed-form variance is used (fully
        differentiable).  For other kernels a differentiable proxy based on
        the distance to the nearest training embedding in the spectral basis is
        returned.  This proxy increases when a query lies far from the training
        manifold and is stable for back-propagation.

        The proxy is computed as

        .. math::

            \\tilde{\\sigma}^2(x) = \\lambda + \\sum_i w_i \\|E(x) - E(x_i)\\|_2^2

        where :math:`w_i = \\text{softmax}(-\\|E(x) - E(x_i)\\|_2^2)` are soft
        attention weights over training embeddings.

        Args:
            x: Query locations of shape ``(m, d)``.
            embedding_model: The embedding network (gradients flow through).
            embeddings: Training embeddings of shape ``(n, D)``.
            kernel_operator: The kernel operator.
            preconditioner: The learned preconditioner.
            alpha: Solution vector of shape ``(n,)``.
            lambda_reg: Regularisation weight.

        Returns:
            Differentiable predictive variance of shape ``(m,)``, clamped
            to be non-negative.
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
        """Return the condition number of the preconditioned system.

        Estimates :math:`\\kappa(P^{-1} K) = \\lambda_{\\max} / \\lambda_{\\min}`
        of the preconditioned kernel operator using power iteration (for
        the largest eigenvalue) and PCG-based inverse iteration (for the
        smallest eigenvalue).

        Args:
            kernel_operator: The fitted kernel operator.
            preconditioner: The learned preconditioner.

        Returns:
            The estimated condition number as a positive float.
        """
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
