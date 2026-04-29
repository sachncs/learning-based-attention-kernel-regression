"""High-level sklearn-compatible LAKER estimator."""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy
import torch
import torch.nn as nn

from laker.backend import get_default_device, get_default_dtype, to_tensor
from laker.embeddings import PositionEmbedding
from laker.kernels import AttentionKernelOperator
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient

logger = logging.getLogger(__name__)


class LAKERRegressor:
    """Learning-based Attention Kernel Regression estimator.

    Fits the regularised attention kernel regression problem

    .. math::
        \\min_\\alpha \\|G \\alpha - y\\|_2^2 + \\lambda \\alpha^\\top G \\alpha

    where ``G = exp(E E^T)`` is an exponential attention kernel induced by
    learned embeddings ``E``. The linear system ``(lambda I + G) alpha = y``
    is solved efficiently using a learned CCCP preconditioner inside PCG.

    The estimator follows the ``scikit-learn`` API with ``fit`` and
    ``predict`` methods.

    Args:
        embedding_dim: Dimension ``d_e`` of the learned embeddings (default 10).
        lambda_reg: Tikhonov regularisation ``lambda`` (default 1e-2).
        gamma: CCCP regularisation parameter (default 1e-1).
        num_probes: Number of random directions ``N_r`` for preconditioner
            learning. ``None`` selects an adaptive heuristic.
        epsilon: Numerical safeguard in CCCP update (default 1e-8).
        base_rho: Base shrinkage parameter (default 0.05).
        cccp_max_iter: Maximum CCCP iterations (default 200).
        cccp_tol: CCCP convergence tolerance (default 1e-6).
        pcg_tol: PCG residual tolerance (default 1e-10).
        pcg_max_iter: Maximum PCG iterations (default 1000).
        chunk_size: Chunk size for matrix-free kernel matvecs. ``None``
            materialises the full kernel for ``n <= 5000``.
        embedding_module: Optional custom ``nn.Module`` mapping locations to
            embeddings. If ``None``, ``PositionEmbedding`` is used.
        device: torch device (``"cuda"``, ``"cpu"``, etc.).
        dtype: torch dtype (``torch.float32`` or ``torch.float64``).
        verbose: Whether to log training progress.

    Raises:
        ValueError: If any hyperparameter is out of its valid range.
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
        pcg_tol: float = 1e-10,
        pcg_max_iter: int = 1000,
        chunk_size: Optional[int] = None,
        embedding_module: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = True,
    ) -> None:
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
        self.verbose = verbose

        if device is None:
            device = get_default_device()
        elif isinstance(device, str):
            device = torch.device(device)
        if dtype is None:
            dtype = get_default_dtype()
        self.device = device
        self.dtype = dtype

        # Fitted attributes
        self.embeddings: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None
        self.preconditioner: Optional[CCCPPreconditioner] = None
        self.kernel_operator: Optional[AttentionKernelOperator] = None
        self.embedding_model: Optional[nn.Module] = None

    def fit(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
    ) -> "LAKERRegressor":
        """Fit the LAKER model to sparse measurements.

        Args:
            x: Measurement locations of shape ``(n, dx)``.
            y: Noisy observations of shape ``(n,)`` or ``(n, 1)``.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If ``x`` is not 2-D or ``y`` is not 1-D.
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        y = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()

        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        n = x.shape[0]
        input_dim = x.shape[1]

        if self.verbose:
            logger.info("Fitting LAKER on n=%d, dx=%d", n, input_dim)

        # 1. Build or use provided embeddings
        if self.embedding_module is not None:
            self.embedding_model = self.embedding_module.to(self.device)
            with torch.no_grad():
                self.embeddings = self.embedding_model(x)
        else:
            self.embedding_model = PositionEmbedding(
                input_dim=input_dim,
                embedding_dim=self.embedding_dim,
                device=self.device,
                dtype=self.dtype,
            )
            with torch.no_grad():
                self.embeddings = self.embedding_model(x)

        # Auto-select chunk size for large problems
        chunk_size = self.chunk_size
        if chunk_size is None and n > 5000:
            chunk_size = max(1024, min(n // 10, 8192))
            if self.verbose:
                logger.info("Auto-selected chunk_size=%d for n=%d", chunk_size, n)

        # 2. Build attention kernel operator
        self.kernel_operator = AttentionKernelOperator(
            embeddings=self.embeddings,
            lambda_reg=self.lambda_reg,
            chunk_size=chunk_size,
            device=self.device,
            dtype=self.dtype,
        )

        # 3. Learn CCCP preconditioner
        self.preconditioner = CCCPPreconditioner(
            num_probes=self.num_probes,
            gamma=self.gamma,
            epsilon=self.epsilon,
            base_rho=self.base_rho,
            max_iter=self.cccp_max_iter,
            tol=self.cccp_tol,
            verbose=self.verbose,
            device=self.device,
            dtype=self.dtype,
        )
        self.preconditioner.build(self.kernel_operator.matvec, n)

        # 4. Solve with PCG
        pcg = PreconditionedConjugateGradient(
            tol=self.pcg_tol,
            max_iter=self.pcg_max_iter,
            verbose=self.verbose,
        )
        self.alpha = pcg.solve(
            operator=self.kernel_operator.matvec,
            preconditioner=self.preconditioner.apply,
            rhs=y,
        )

        if self.verbose:
            final_res = (
                torch.linalg.norm(self.kernel_operator.matvec(self.alpha) - y).item()
                / torch.linalg.norm(y).item()
            )
            logger.info(
                "LAKER fit complete: PCG iters=%d, final rel_res=%.3e",
                pcg.iterations,
                final_res,
            )
        return self

    def predict(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
    ) -> torch.Tensor:
        """Reconstruct the radio field at query locations.

        Evaluates ``\\hat{r}(x) = \\sum_i G(x, x_i) \\alpha_i``.

        Args:
            x: Query locations of shape ``(m, dx)``.

        Returns:
            Predictions of shape ``(m,)``.

        Raises:
            RuntimeError: If the model has not been fitted or no embedding model
                is available.
            ValueError: If ``x`` is not 2-D.
        """
        if self.alpha is None or self.embeddings is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")

        with torch.no_grad():
            if self.embedding_model is None:
                raise RuntimeError(
                    "No embedding model available. Ensure the model was fitted with "
                    "an embedding_module or the default PositionEmbedding."
                )
            query_embeddings = self.embedding_model(x)

            # Evaluate kernel between queries and training points
            k_query = self.kernel_operator.kernel_eval(query_embeddings, self.embeddings)
            return k_query @ self.alpha

    def score(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
    ) -> float:
        """Compute negative RMSE as a sklearn-style score.

        Args:
            x: Test locations of shape ``(m, dx)``.
            y: Ground-truth values of shape ``(m,)``.

        Returns:
            Negative RMSE (higher is better).
        """
        y_pred = self.predict(x)
        y_true = to_tensor(y, device=self.device, dtype=self.dtype).squeeze()
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
        return -rmse

    def condition_number(self) -> float:
        """Return the condition number of the preconditioned system.

        Estimates the condition number ``kappa(P (lambda I + G))`` via
        power iteration for the largest eigenvalue and inverse power
        iteration for the smallest.

        Requires the preconditioner to have been built.

        Returns:
            Estimated condition number.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.preconditioner is None or self.kernel_operator is None:
            raise RuntimeError("Model has not been fitted.")
        # Power iteration on P A and (P A)^{-1} to estimate condition number
        n = self.kernel_operator.n
        apply_operator = self.kernel_operator.matvec
        apply_precond = self.preconditioner.apply

        def preconditioned_operator(v: torch.Tensor) -> torch.Tensor:
            """Apply the preconditioned operator P A to a vector."""
            return apply_precond(apply_operator(v))

        # Largest eigenvalue
        v = torch.randn(n, device=self.device, dtype=self.dtype)
        v = v / torch.linalg.norm(v)
        for _ in range(20):
            v = preconditioned_operator(v)
            v = v / torch.linalg.norm(v)
        lam_max = torch.dot(v, preconditioned_operator(v)).item()

        # Smallest eigenvalue via inverse iteration
        v = torch.randn(n, device=self.device, dtype=self.dtype)
        v = v / torch.linalg.norm(v)

        # Use CG to apply (PA)^{-1} approximately.
        # Since PA is well-conditioned by construction, identity preconditioning
        # inside CG is acceptable and avoids circular dependence.
        pcg = PreconditionedConjugateGradient(tol=1e-6, max_iter=200, verbose=False)
        for _ in range(10):
            v = pcg.solve(
                operator=preconditioned_operator,
                preconditioner=lambda x: x,
                rhs=v,
            )
            v = v / torch.linalg.norm(v)
        lam_min = max(torch.dot(v, preconditioned_operator(v)).item(), 1e-16)

        return lam_max / lam_min

    def get_params(self, deep: bool = True) -> dict:
        """Return estimator parameters for sklearn compatibility.

        Args:
            deep: Whether to return parameters of nested objects.

        Returns:
            Dictionary mapping parameter names to values.
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
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "LAKERRegressor":
        """Set estimator parameters for sklearn compatibility.

        Args:
            **params: Parameter names and values.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If an unknown parameter is provided.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key!r} for LAKERRegressor")
            setattr(self, key, value)
        return self

    def save(self, path: str) -> None:
        """Serialize the fitted model to disk.

        Saves all hyperparameters and fitted tensors so that ``load()``
        can reconstruct an identical model.

        Args:
            path: File path ending in ``.pt`` or ``.pth``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.alpha is None:
            raise RuntimeError("Model has not been fitted. Call fit() before saving.")
        state = {
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
            "device": str(self.device),
            "dtype": str(self.dtype),
            "verbose": self.verbose,
            "embeddings": self.embeddings,
            "alpha": self.alpha,
        }
        if self.embedding_model is not None:
            state["embedding_model_state"] = self.embedding_model.state_dict()
            state["embedding_model_class"] = self.embedding_model.__class__.__name__
            # Store input_dim for correct reconstruction
            if hasattr(self.embedding_model, "input_dim"):
                state["input_dim"] = self.embedding_model.input_dim
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "LAKERRegressor":
        """Deserialize a model from disk.

        Args:
            path: File path previously passed to ``save()``.

        Returns:
            Reconstructed ``LAKERRegressor`` instance ready for ``predict()``.
        """
        state = torch.load(path, weights_only=False)
        dtype = torch.float32 if "float32" in state["dtype"] else torch.float64
        model = cls(
            embedding_dim=state["embedding_dim"],
            lambda_reg=state["lambda_reg"],
            gamma=state["gamma"],
            num_probes=state["num_probes"],
            epsilon=state["epsilon"],
            base_rho=state["base_rho"],
            cccp_max_iter=state["cccp_max_iter"],
            cccp_tol=state["cccp_tol"],
            pcg_tol=state["pcg_tol"],
            pcg_max_iter=state["pcg_max_iter"],
            chunk_size=state.get("chunk_size"),
            device=state["device"],
            dtype=dtype,
            verbose=state["verbose"],
        )
        model.embeddings = state["embeddings"].to(model.device)
        model.alpha = state["alpha"].to(model.device)
        if "embedding_model_state" in state:
            from laker.embeddings import PositionEmbedding

            input_dim = state.get("input_dim", 2)
            model.embedding_model = PositionEmbedding(
                input_dim=input_dim,
                embedding_dim=model.embedding_dim,
                device=model.device,
                dtype=dtype,
            )
            model.embedding_model.load_state_dict(state["embedding_model_state"])
        model.kernel_operator = AttentionKernelOperator(
            embeddings=model.embeddings,
            lambda_reg=model.lambda_reg,
            chunk_size=model.chunk_size,
            device=model.device,
            dtype=dtype,
        )
        return model
