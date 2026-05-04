"""Streaming updates and regularisation-path fitting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from laker.core import LAKERCore
    from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class StreamingUpdater:
    """Incremental updates and continuation-path fitting."""

    def __init__(self, core: "LAKERCore") -> None:
        """Initialise the streaming updater."""
        self.core = core

    def partial_fit(
        self,
        regressor: "LAKERRegressor",
        x_new: torch.Tensor,
        y_new: torch.Tensor,
        forgetting_factor: float = 1.0,
        rebuild_threshold: int = 100,
    ) -> "LAKERRegressor":
        """Update the model with one or more new observations."""
        if regressor.alpha is None or regressor.embeddings is None:
            raise RuntimeError("Model has not been fitted. Call fit() before partial_fit().")

        if x_new.dim() != 2:
            raise ValueError(f"x_new must be 2-D, got shape {x_new.shape}")
        if y_new.dim() != 1:
            raise ValueError(f"y_new must be 1-D, got shape {y_new.shape}")

        m = x_new.shape[0]
        total_new = getattr(regressor, "partial_fit_count", 0) + m

        if total_new >= rebuild_threshold:
            regressor.partial_fit_count = 0
            raise RuntimeError(
                "partial_fit rebuild threshold exceeded. "
                "Please concatenate all data and call fit() for a full refit."
            )

        regressor.partial_fit_count = total_new

        embedded_input = x_new.to(dtype=self.core.embedding_dtype)
        with torch.no_grad():
            new_embeddings = regressor.embedding_model(embedded_input)
        if self.core.embedding_dtype != self.core.dtype:
            new_embeddings = new_embeddings.to(dtype=self.core.dtype)

        old_n = regressor.embeddings.shape[0]
        regressor.embeddings = torch.cat([regressor.embeddings, new_embeddings], dim=0)

        regressor.kernel_operator = self.core.build_kernel_operator(regressor.embeddings)

        old_alpha = regressor.alpha * forgetting_factor
        y_old = getattr(regressor, "y_train", None)
        if y_old is None:
            y_old = torch.zeros(old_n, device=self.core.device, dtype=self.core.dtype)
        y_extended = torch.cat([y_old, y_new])

        x0 = torch.cat(
            [
                old_alpha,
                torch.zeros(m, device=self.core.device, dtype=self.core.dtype),
            ]
        )

        regressor.preconditioner = self.core.build_preconditioner(
            regressor.kernel_operator.matvec,
            regressor.embeddings.shape[0],
            diagonal=regressor.kernel_operator.diagonal(),
        )

        regressor.alpha, regressor.pcg_iterations_ = self.core.solve_pcg(
            regressor.kernel_operator,
            regressor.preconditioner,
            y_extended,
            x0=x0,
        )

        regressor.y_train = y_extended
        if hasattr(regressor, "x_train") and regressor.x_train is not None:
            regressor.x_train = torch.cat([regressor.x_train, x_new], dim=0)

        if self.core.verbose:
            logger.info(
                "partial_fit: added %d points, total=%d, PCG iters=%d",
                m,
                regressor.embeddings.shape[0],
                regressor.pcg_iterations_,
            )
        return regressor

    def fit_path(
        self,
        regressor: "LAKERRegressor",
        x: torch.Tensor,
        y: torch.Tensor,
        lambda_reg_grid: list[float],
        reuse_precond: bool = True,
    ) -> dict:
        """Fit a regularization path over a sequence of lambda_reg values."""
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if not lambda_reg_grid:
            raise ValueError("lambda_reg_grid must not be empty")

        embeddings, model = self.core.compute_embeddings(x)
        n = embeddings.shape[0]

        sorted_lambdas = sorted(lambda_reg_grid, reverse=True)

        alphas = []
        pcg_iters = []
        rel_reses = []
        x0 = None
        precond = None

        for lambda_reg_value in sorted_lambdas:
            kernel_op = self.core.build_kernel_operator(embeddings, lambda_reg=lambda_reg_value)
            if precond is None or not reuse_precond:
                precond = self.core.build_preconditioner(
                    kernel_op.matvec,
                    n,
                    gamma=self.core.gamma,
                    num_probes=self.core.num_probes,
                    diagonal=kernel_op.diagonal(),
                )
            alpha, iters = self.core.solve_pcg(kernel_op, precond, y, x0=x0)
            alphas.append(alpha)
            pcg_iters.append(iters)
            final_res = (
                torch.linalg.norm(kernel_op.matvec(alpha) - y).item() / torch.linalg.norm(y).item()
            )
            rel_reses.append(final_res)
            x0 = alpha.clone()

        # Store final state on regressor for prediction
        regressor.embeddings = embeddings
        regressor.embedding_model = model
        regressor.kernel_operator = self.core.build_kernel_operator(
            embeddings, lambda_reg=sorted_lambdas[-1]
        )
        regressor.preconditioner = self.core.build_preconditioner(
            regressor.kernel_operator.matvec,
            n,
            gamma=self.core.gamma,
            num_probes=self.core.num_probes,
            diagonal=regressor.kernel_operator.diagonal(),
        )
        regressor.alpha = alphas[-1]
        regressor.pcg_iterations_ = pcg_iters[-1]
        regressor.y_train = y

        path = {
            "lambda_reg": sorted_lambdas,
            "alphas": alphas,
            "pcg_iters": pcg_iters,
            "final_rel_res": rel_reses,
        }
        regressor.path_ = path
        return path

    def fit_continuation(
        self,
        regressor: "LAKERRegressor",
        x: torch.Tensor,
        y: torch.Tensor,
        lambda_max: Optional[float] = None,
        lambda_min: Optional[float] = None,
        n_stages: int = 5,
        reuse_precond: bool = True,
    ) -> "LAKERRegressor":
        """Fit with a continuation schedule over decreasing regularisation."""
        if lambda_max is None:
            lambda_max = 10.0 * self.core.lambda_reg
        if lambda_min is None:
            lambda_min = self.core.lambda_reg
        if n_stages < 1:
            raise ValueError(f"n_stages must be positive, got {n_stages}")
        if lambda_max <= 0 or lambda_min <= 0:
            raise ValueError("lambda_max and lambda_min must be positive")

        ratio = (lambda_min / lambda_max) ** (1.0 / max(1, n_stages - 1))
        schedule = [lambda_max * (ratio**k) for k in range(n_stages)]
        schedule[-1] = lambda_min

        path = self.fit_path(
            regressor,
            x,
            y,
            lambda_reg_grid=schedule,
            reuse_precond=reuse_precond,
        )
        regressor.lambda_reg = float(lambda_min)
        regressor.pcg_iterations_ = path["pcg_iters"][-1]
        return regressor
