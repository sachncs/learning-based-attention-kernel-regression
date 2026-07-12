"""High-level sklearn-compatible LAKER estimator.

This module provides :class:`LAKERRegressor`, a scikit-learn-compatible
regressor that wraps the full LAKER pipeline.  It follows the standard
``fit`` / ``predict`` / ``score`` API and can be used with scikit-learn
utilities such as :class:`~sklearn.model_selection.GridSearchCV` and
:class:`~sklearn.pipeline.Pipeline`.

The regressor delegates all heavy computation to
:class:`~laker.core.LAKERCore` (kernel construction, preconditioning, PCG
solve) and to specialised helper classes:

* :class:`~laker.training.EmbeddingTrainer` -- end-to-end embedding
  training, residual correction, bilevel learning, uncertainty-aware
  training.
* :class:`~laker.search.HyperparameterSearch` -- grid search and Bayesian
  optimisation over hyperparameters.
* :class:`~laker.streaming.StreamingUpdater` -- incremental updates,
  regularization paths, and continuation methods.
* :class:`~laker.persistence.ModelPersistence` -- serialisation to disk.

**Quick start**::

    from laker.models import LAKERRegressor

    reg = LAKERRegressor(embedding_dim=16, lambda_reg=0.01)
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    print(reg.score(x_test, y_test))   # negative RMSE

**Serialization**::

    reg.save("model.pt")
    loaded = LAKERRegressor.load("model.pt")
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy
import torch
import torch.nn as nn

from laker.backend import to_tensor
from laker.core import LAKERCore
from laker.kernels import KernelOperator
from laker.persistence import ModelPersistence
from laker.preconditioner import AdaptivePreconditioner, CCCPPreconditioner
from laker.search import HyperparameterSearch
from laker.streaming import StreamingUpdater
from laker.training import EmbeddingTrainer

logger = logging.getLogger(__name__)


class LAKERRegressor:
    r"""Learning-based Attention Kernel Regression estimator.

    Fits the regularised attention kernel regression problem

    .. math::
        \min_\alpha \|G \alpha - y\|_2^2 + \lambda \alpha^\top G \alpha

    where ``G = exp(E E^T)`` is an exponential attention kernel induced by
    learned embeddings ``E``. The linear system ``(lambda I + G) alpha = y``
    is solved efficiently using a learned CCCP preconditioner inside PCG.

    The estimator follows the ``scikit-learn`` API with ``fit`` and
    ``predict`` methods.

    Args:
        embedding_dim: Dimensionality of the embedding space.
        lambda_reg: Regularisation weight :math:`\lambda` in the kernel
            ridge regression objective.
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
        embedding_module: Optional pre-built ``nn.Module`` embedding network.
        kernel_approx: Kernel approximation method.  One of ``None`` (exact),
            ``"nystrom"``, ``"rff"``, ``"knn"``, ``"ski"``, ``"twoscale"``,
            or ``"spectral"``.
        num_landmarks: Number of landmarks for Nyström / two-scale kernels.
        num_features: Number of random Fourier features for RFF kernel.
        k_neighbors: Number of nearest neighbours for sparse k-NN kernel.
        grid_size: Grid resolution for SKI kernel.
        distributed: Whether to use multi-device distributed kernel.
        twoscale_alpha: Blending weight for the two-scale kernel.
        landmark_method: Landmark selection strategy (``"greedy"`` or
            ``"leverage"``).
        landmark_pilot_size: Pilot size for leverage-score landmarks.
        spectral_knots: Number of spline knots for spectral kernel.
        preconditioner: Preconditioner type (``"cccp"`` or ``"adaptive"``).
        embedding_dtype: Dtype for embedding computation.
        device: Target PyTorch device.
        dtype: Target floating-point dtype.
        verbose: Whether to emit diagnostic log messages.
        residual_corrector: Optional pre-built residual corrector module.

    Attributes:
        embeddings: Fitted training embeddings of shape ``(n, D)``, or
            ``None`` before :meth:`fit`.
        alpha: Solution vector of shape ``(n,)``, or ``None`` before fit.
        kernel_operator: The fitted kernel operator, or ``None`` before fit.
        preconditioner: The learned preconditioner, or ``None`` before fit.
        embedding_model: The embedding network, or ``None`` before fit.
        x_train: Training locations stored during :meth:`fit`.
        y_train: Training targets stored during :meth:`fit`.

    Examples:
        Basic usage::

            from laker.models import LAKERRegressor
            reg = LAKERRegressor(embedding_dim=16, lambda_reg=0.01)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)

        With RFF approximation::

            reg = LAKERRegressor(kernel_approx="rff", num_features=128)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            sigma = reg.predict_variance(x_test)

        Serialization::

            reg.save("model.pt")
            loaded = LAKERRegressor.load("model.pt")
    """

    # Hyperparameters delegated to self.core
    HYPERPARAMS = (
        "embedding_dim",
        "lambda_reg",
        "gamma",
        "num_probes",
        "epsilon",
        "base_rho",
        "cccp_max_iter",
        "cccp_tol",
        "pcg_tol",
        "pcg_max_iter",
        "chunk_size",
        "kernel_approx",
        "num_landmarks",
        "num_features",
        "k_neighbors",
        "grid_size",
        "distributed",
        "twoscale_alpha",
        "landmark_method",
        "landmark_pilot_size",
        "spectral_knots",
        "preconditioner_strategy",
        "embedding_dtype",
        "device",
        "dtype",
        "verbose",
        "embedding_module",
    )

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
        preconditioner: str = "cccp",
        embedding_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = True,
        residual_corrector: Optional[nn.Module] = None,
    ) -> None:
        """Initialise the LAKER regressor.

        All hyperparameters are validated and forwarded to an internal
        :class:`~laker.core.LAKERCore` instance.  Helper objects
        (trainer, search, streaming, persistence) are also created.

        Args:
            embedding_dim: Dimensionality of the embedding space.
            lambda_reg: Regularisation weight.
            gamma: Kernel bandwidth for the CCCP preconditioner.
            num_probes: Number of probe vectors for the preconditioner.
            epsilon: Numerical stability constant.
            base_rho: Base spectral norm bound for CCCP.
            cccp_max_iter: Maximum CCCP iterations.
            cccp_tol: CCCP convergence tolerance.
            pcg_tol: PCG relative residual tolerance.
            pcg_max_iter: Maximum PCG iterations.
            chunk_size: Tile size for chunked kernel evaluations.
            embedding_module: Optional pre-built embedding module.
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
            preconditioner: ``"cccp"`` or ``"adaptive"``.
            embedding_dtype: Dtype for embedding computation.
            device: Target PyTorch device.
            dtype: Target floating-point dtype.
            verbose: Whether to emit diagnostic log messages.
            residual_corrector: Optional pre-built residual corrector
                module.

        Raises:
            ValueError: If any hyperparameter is out of its valid range.
        """
        # --- validation (exactly as before) --------------------------------
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if lambda_reg <= 0:
            raise ValueError(f"lambda_reg must be positive, got {lambda_reg}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not 0 <= base_rho <= 1:
            raise ValueError(f"base_rho must be in [0, 1], got {base_rho}")
        if cccp_max_iter <= 0:
            raise ValueError(f"cccp_max_iter must be positive, got {cccp_max_iter}")
        if cccp_tol <= 0:
            raise ValueError(f"cccp_tol must be positive, got {cccp_tol}")
        if pcg_tol <= 0:
            raise ValueError(f"pcg_tol must be positive, got {pcg_tol}")
        if pcg_max_iter <= 0:
            raise ValueError(f"pcg_max_iter must be positive, got {pcg_max_iter}")
        if kernel_approx not in (
            None,
            "nystrom",
            "rff",
            "knn",
            "ski",
            "twoscale",
            "spectral",
        ):
            raise ValueError(
                "kernel_approx must be None, 'nystrom', 'rff', 'knn', 'ski', "
                f"'twoscale', or 'spectral', got {kernel_approx}"
            )
        if k_neighbors is not None and k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
        if grid_size is not None and grid_size < 2:
            raise ValueError(f"grid_size must be at least 2, got {grid_size}")
        if not 0.0 <= twoscale_alpha <= 1.0:
            raise ValueError(f"twoscale_alpha must be in [0, 1], got {twoscale_alpha}")
        if landmark_method not in ("greedy", "leverage"):
            raise ValueError(
                f"landmark_method must be 'greedy' or 'leverage', got {landmark_method}"
            )
        if landmark_pilot_size <= 0:
            raise ValueError(f"landmark_pilot_size must be positive, got {landmark_pilot_size}")
        if preconditioner not in ("cccp", "adaptive"):
            raise ValueError(f"preconditioner must be 'cccp' or 'adaptive', got {preconditioner}")
        if spectral_knots <= 0:
            raise ValueError(f"spectral_knots must be positive, got {spectral_knots}")

        # --- helpers (bypass __setattr__) -----------------------------------
        object.__setattr__(
            self,
            "core",
            LAKERCore(
                embedding_dim=embedding_dim,
                lambda_reg=lambda_reg,
                gamma=gamma,
                num_probes=num_probes,
                epsilon=epsilon,
                base_rho=base_rho,
                cccp_max_iter=cccp_max_iter,
                cccp_tol=cccp_tol,
                pcg_tol=pcg_tol,
                pcg_max_iter=pcg_max_iter,
                chunk_size=chunk_size,
                embedding_module=embedding_module,
                kernel_approx=kernel_approx,
                num_landmarks=num_landmarks,
                num_features=num_features,
                k_neighbors=k_neighbors,
                grid_size=grid_size,
                distributed=distributed,
                twoscale_alpha=twoscale_alpha,
                landmark_method=landmark_method,
                landmark_pilot_size=landmark_pilot_size,
                spectral_knots=spectral_knots,
                preconditioner_strategy=preconditioner,
                embedding_dtype=embedding_dtype,
                device=device,
                dtype=dtype,
                verbose=verbose,
            ),
        )
        object.__setattr__(self, "search", HyperparameterSearch(self.core))
        object.__setattr__(self, "streaming", StreamingUpdater(self.core))
        object.__setattr__(self, "trainer", EmbeddingTrainer(self.core))
        object.__setattr__(self, "persistence", ModelPersistence())

        # --- fitted state ---------------------------------------------------
        self.embeddings: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None
        self.kernel_operator: Optional[KernelOperator] = None
        self.preconditioner: Optional[Union[CCCPPreconditioner, AdaptivePreconditioner]] = None
        self.embedding_model: Optional[nn.Module] = None
        self.residual_corrector = residual_corrector
        self.x_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None
        self.partial_fit_count: int = 0

    # ------------------------------------------------------------------
    # Hyperparameter delegation
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """Delegate hyperparameter access to the core.

        If ``name`` is one of the registered hyperparameters, the
        corresponding attribute is read from the internal
        :class:`~laker.core.LAKERCore` instance.  This ensures that
        hyperparameters set via ``get_params`` / ``set_params`` are
        always in sync with the core pipeline.

        Args:
            name: Attribute name to look up.

        Returns:
            The value of the hyperparameter from the core.

        Raises:
            AttributeError: If ``name`` is not a recognised hyperparameter.
        """
        if name in self.HYPERPARAMS:
            return getattr(self.core, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate hyperparameter writes to the core.

        When ``name`` is a registered hyperparameter and the core has
        already been initialised, the value is written directly to the
        internal :class:`~laker.core.LAKERCore` instance.  String values
        for ``device``, ``dtype``, and ``embedding_dtype`` are converted
        to the appropriate ``torch`` types.

        Args:
            name: Attribute name to set.
            value: New value for the attribute.
        """
        if name in self.HYPERPARAMS and "core" in self.__dict__:
            if name == "device" and isinstance(value, str):
                value = torch.device(value)
            if name == "dtype" and isinstance(value, str):
                value = torch.float32 if "float32" in value else torch.float64
            if name == "embedding_dtype" and isinstance(value, str):
                value = torch.float32 if "float32" in value else torch.float64
            setattr(self.core, name, value)
        else:
            super().__setattr__(name, value)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def fit(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        x0: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> "LAKERRegressor":
        """Fit the LAKER model to sparse measurements.

        Executes the full pipeline:

        1. Compute embeddings :math:`E = f_{\\text{embed}}(X)`.
        2. Build the attention kernel operator
           :math:`K = \\exp(E E^\\top) + \\lambda I`.
        3. Learn the CCCP/adaptive preconditioner.
        4. Solve :math:`(K + \\lambda I)\\alpha = y` via PCG.

        The fitted state (``embeddings``, ``alpha``, ``kernel_operator``,
        ``preconditioner``, ``embedding_model``) is stored on the
        regressor for subsequent calls to :meth:`predict` and
        :meth:`predict_variance`.

        Args:
            x: Training locations of shape ``(n, d)``.  Accepts a
                ``torch.Tensor`` or ``numpy.ndarray``.
            y: Training observations of shape ``(n,)``.
            x0: Optional initial guess for the PCG solver.
            seed: Random seed for reproducibility of the preconditioner.

        Returns:
            ``self``, to allow method chaining.

        Raises:
            ValueError: If ``x`` is not 2-D or ``y`` is not 1-D.

        Examples:
            >>> reg = LAKERRegressor(lambda_reg=0.01)
            >>> reg.fit(x_train, y_train)
            >>> reg.predict(x_test)
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()

        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        self.x_train = x
        self.y_train = y
        self.embeddings, self.embedding_model = self.core.compute_embeddings(x)
        kernel_operator = self.core.build_kernel_operator(self.embeddings)
        self.kernel_operator = kernel_operator
        preconditioner = self.core.build_preconditioner(
            kernel_operator.matvec,
            self.embeddings.shape[0],
            seed=seed,
            diagonal=kernel_operator.diagonal(),
        )
        self.preconditioner = preconditioner
        self.alpha, self.pcg_iterations_ = self.core.solve_pcg(
            kernel_operator, preconditioner, y, x0=x0
        )
        return self

    def predict(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
    ) -> torch.Tensor:
        """Reconstruct the radio field at query locations.

        Evaluates the kernel interpolation
        :math:`\\hat{f}(x) = k(x, X)^\\top \\alpha` at each query point,
        optionally adding the residual corrector output.

        Args:
            x: Query locations of shape ``(m, d)``.  Accepts a
                ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Predicted field values of shape ``(m,)``.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If ``x`` is not 2-D.

        Examples:
            >>> y_pred = reg.predict(x_test)
        """
        if self.alpha is None or self.embeddings is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")

        return self.core.predict(
            x,
            self.embedding_model,
            self.embeddings,
            self.kernel_operator,
            self.alpha,
            self.residual_corrector,
        )

    def predict_variance(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
    ) -> torch.Tensor:
        """Predictive variance (uncertainty) at query locations.

        Computes the posterior variance
        :math:`\\sigma^2(x) = k(x,x) - k(x,X)^\\top (K+\\lambda I)^{-1} k(x,X)`
        for each query point.

        Args:
            x: Query locations of shape ``(m, d)``.  Accepts a
                ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Predictive variance of shape ``(m,)``, clamped to be
            non-negative.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If ``x`` is not 2-D.

        Examples:
            >>> sigma = reg.predict_variance(x_test)
        """
        if self.alpha is None or self.embeddings is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")

        return self.core.predict_variance(
            x,
            self.embedding_model,
            self.embeddings,
            self.kernel_operator,
            self.preconditioner,
            self.alpha,
            self.lambda_reg,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def fit_with_search(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        val_fraction: float = 0.2,
        lambda_reg_grid: Optional[list[float]] = None,
        gamma_grid: Optional[list[float]] = None,
        num_probes_grid: Optional[list[int]] = None,
        warm_start: bool = True,
    ) -> "LAKERRegressor":
        """Fit with validation-based grid search over key hyperparameters."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        return self.search.fit_with_search(
            self,
            x,
            y,
            val_fraction,
            lambda_reg_grid,
            gamma_grid,
            num_probes_grid,
            warm_start,
        )

    def fit_with_bo(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        val_fraction: float = 0.2,
        n_calls: int = 15,
        n_initial_points: int = 5,
        lambda_reg_bounds: tuple[float, float] = (1e-4, 1.0),
        gamma_bounds: tuple[float, float] = (0.0, 2.0),
        num_probes_bounds: tuple[int, int] = (20, 300),
    ) -> "LAKERRegressor":
        """Fit with Bayesian Optimisation over key hyperparameters."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        return self.search.fit_with_bo(
            self,
            x,
            y,
            val_fraction,
            n_calls,
            n_initial_points,
            lambda_reg_bounds,
            gamma_bounds,
            num_probes_bounds,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def partial_fit(
        self,
        x_new: Union[torch.Tensor, "numpy.ndarray"],
        y_new: Union[torch.Tensor, "numpy.ndarray"],
        forgetting_factor: float = 1.0,
        rebuild_threshold: int = 100,
    ) -> "LAKERRegressor":
        """Update the model with one or more new observations."""
        x_new = to_tensor(x_new, device=self.device, dtype=self.dtype)
        y_new = to_tensor(y_new, device=self.device, dtype=self.dtype).squeeze()
        return self.streaming.partial_fit(self, x_new, y_new, forgetting_factor, rebuild_threshold)

    def fit_path(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        lambda_reg_grid: list[float],
        reuse_precond: bool = True,
    ) -> dict:
        """Fit a regularization path over a sequence of lambda_reg values."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        return self.streaming.fit_path(self, x, y, lambda_reg_grid, reuse_precond)

    def fit_continuation(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        lambda_max: Optional[float] = None,
        lambda_min: Optional[float] = None,
        n_stages: int = 5,
        reuse_precond: bool = True,
    ) -> "LAKERRegressor":
        """Fit with a continuation schedule over decreasing regularisation."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        return self.streaming.fit_continuation(
            self, x, y, lambda_max, lambda_min, n_stages, reuse_precond
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit_learned_embeddings(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        lr: float = 1e-3,
        epochs: int = 50,
        rebuild_freq: int = 10,
        patience: int = 5,
    ) -> "LAKERRegressor":
        """Optimise the embedding MLP weights end-to-end on the regression objective."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        return self.trainer.fit_learned_embeddings(self, x, y, lr, epochs, rebuild_freq, patience)

    def fit_residual_corrector(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        val_fraction: float = 0.2,
        epochs: int = 200,
        patience: int = 10,
        weight_decay: float = 1e-2,
        lr: float = 1e-3,
    ) -> "LAKERRegressor":
        """Train a small residual corrector on ``y - y_hat_laker``."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        return self.trainer.fit_residual_corrector(
            self, x, y, val_fraction, epochs, patience, weight_decay, lr
        )

    def fit_bilevel(
        self,
        x_train: Union[torch.Tensor, "numpy.ndarray"],
        y_train: Union[torch.Tensor, "numpy.ndarray"],
        x_val: Union[torch.Tensor, "numpy.ndarray"],
        y_val: Union[torch.Tensor, "numpy.ndarray"],
        lr: float = 1e-3,
        epochs: int = 20,
        patience: int = 5,
    ) -> "LAKERRegressor":
        """Optimise hyperparameters via bilevel learning with implicit differentiation."""
        x_train = to_tensor(x_train, device=self.device, dtype=self.dtype)
        y_train = to_tensor(y_train, device=self.device, dtype=self.dtype).squeeze()
        x_val = to_tensor(x_val, device=self.device, dtype=self.dtype)
        y_val = to_tensor(y_val, device=self.device, dtype=self.dtype).squeeze()
        return self.trainer.fit_bilevel(self, x_train, y_train, x_val, y_val, lr, epochs, patience)

    def fit_uncertainty_aware(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        lr: float = 1e-3,
        epochs: int = 50,
        beta: float = 0.1,
        variance_subset: float = 0.2,
        patience: int = 5,
    ) -> "LAKERRegressor":
        """Train embeddings with a negative log-likelihood + calibration objective.

        Minimises::

            L = NLL(y | mu, sigma^2) + beta * calibration_penalty

        where ``mu`` and ``sigma^2`` are the LAKER predictive mean and variance.
        This prevents overconfident predictions and improves uncertainty
        quantification for active sensing and sensor placement.
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        return self.trainer.fit_uncertainty_aware(
            self, x, y, lr, epochs, beta, variance_subset, patience
        )

    # ------------------------------------------------------------------
    # Scoring / diagnostics
    # ------------------------------------------------------------------
    def score(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
    ) -> float:
        """Compute negative RMSE as a sklearn-style score.

        Higher is better (consistent with the scikit-learn convention
        where ``score`` returns a quantity to be maximised).

        Args:
            x: Evaluation locations of shape ``(m, d)``.
            y: True observations of shape ``(m,)``.

        Returns:
            The negative root mean squared error, i.e.
            :math:`-\\sqrt{\\frac{1}{m}\\|\\hat{y} - y\\|_2^2}`.
        """
        y_pred = self.predict(x)
        y_true = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
        return -rmse

    def score_r2(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
    ) -> float:
        """Compute the coefficient of determination R^2.

        .. math::

            R^2 = 1 - \\frac{\\sum_i (y_i - \\hat{y}_i)^2}
                        {\\sum_i (y_i - \\bar{y})^2}

        Returns ``1.0`` for a perfect fit, ``0.0`` when the model always
        predicts the mean, and negative values when the model is worse
        than the mean.

        Args:
            x: Evaluation locations of shape ``(m, d)``.
            y: True observations of shape ``(m,)``.

        Returns:
            The :math:`R^2` score as a float.
        """
        y_pred = self.predict(x)
        y_true = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    def condition_number(self) -> float:
        """Return the condition number of the preconditioned system."""
        if self.preconditioner is None or self.kernel_operator is None:
            raise RuntimeError("Model has not been fitted.")
        return self.core.condition_number(self.kernel_operator, self.preconditioner)

    # ------------------------------------------------------------------
    # Sklearn compatibility
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        """Return estimator parameters for sklearn compatibility.

        Returns a dictionary of all hyperparameters that can be passed
        to the constructor.  This method is required for compatibility
        with scikit-learn's cross-validation and model-selection
        utilities.

        Args:
            deep: If ``True``, also returns the parameters of contained
                sub-objects (currently unused but accepted for API
                compatibility).

        Returns:
            Dictionary mapping parameter names to their values.
        """
        return {
            "embedding_dim": self.embedding_dim,
            "lambda_reg": self.lambda_reg,
            "gamma": self.gamma,
            "num_probes": self.num_probes,
            "epsilon": self.epsilon,
            "base_rho": self.base_rho,
            "cccp_max_iter": self.cccp_max_iter,
            "cccp_tol": self.cccp_tol,
            "pcg_tol": self.pcg_tol,
            "pcg_max_iter": self.pcg_max_iter,
            "chunk_size": self.chunk_size,
            "embedding_module": self.embedding_module,
            "kernel_approx": self.kernel_approx,
            "num_landmarks": self.num_landmarks,
            "num_features": self.num_features,
            "k_neighbors": self.k_neighbors,
            "grid_size": self.grid_size,
            "distributed": self.distributed,
            "twoscale_alpha": self.twoscale_alpha,
            "landmark_method": self.landmark_method,
            "landmark_pilot_size": self.landmark_pilot_size,
            "preconditioner": self.preconditioner_strategy,
            "embedding_dtype": (
                str(self.embedding_dtype).replace("torch.", "") if self.embedding_dtype else None
            ),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "verbose": self.verbose,
            "residual_corrector": self.residual_corrector,
        }

    def set_params(self, **params) -> "LAKERRegressor":
        """Set estimator parameters for sklearn compatibility.

        Updates one or more hyperparameters by name.  This method is
        required for compatibility with scikit-learn's cross-validation
        and model-selection utilities.

        Args:
            **params: Hyperparameter names and their new values.

        Returns:
            ``self``, to allow method chaining.

        Raises:
            ValueError: If an unrecognized parameter name is provided.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key!r} for LAKERRegressor")
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize the fitted model to disk.

        Delegates to :meth:`ModelPersistence.save`.  The model must be
        fitted before saving.

        Args:
            path: Filesystem path where the model will be saved.
                Overwrites any existing file at this location.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self.persistence.save(self, path)

    @classmethod
    def load(cls, path: str) -> "LAKERRegressor":
        """Deserialize a model from disk.

        Delegates to :meth:`ModelPersistence.load`.  Returns a fully
        reconstructed ``LAKERRegressor`` with all hyperparameters, fitted
        tensors, embedding model, and residual corrector restored.

        Args:
            path: Filesystem path to the serialized model file.

        Returns:
            A new :class:`LAKERRegressor` instance ready for prediction.

        Examples:
            >>> reg = LAKERRegressor.load("model.pt")
            >>> reg.predict(x_test)
        """
        return ModelPersistence.load(path)
