"""Bilevel hyperparameter learning via implicit differentiation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

import torch

from laker.implicit_diff import hypergradient as implicit_hypergradient

if TYPE_CHECKING:
    from laker.core import LAKERCore
    from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class BilevelOptimizer:
    """Outer-loop Adam optimizer over hyperparameters with inner PCG solve.

    The inner problem solves ``(K(theta) + lambda I) alpha = y`` via PCG.
    The outer problem minimises validation loss w.r.t. ``theta`` (e.g.
    ``lambda_reg``, embedding weights) using hypergradients computed by
    implicit differentiation through the fixed-point.
    """

    def __init__(
        self,
        core: "LAKERCore",
        lr: float = 1e-3,
        epochs: int = 20,
        patience: int = 5,
        pcg_tol: float = 1e-6,
        pcg_max_iter: int = 500,
        verbose: bool = True,
    ) -> None:
        """Initialise the bilevel optimiser."""
        self.core = core
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.pcg_tol = pcg_tol
        self.pcg_max_iter = pcg_max_iter
        self.verbose = verbose

    def fit_bilevel(
        self,
        regressor: "LAKERRegressor",
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        hyperparameters: Optional[List[torch.Tensor]] = None,
    ) -> "LAKERRegressor":
        """Optimise hyperparameters via bilevel learning.

        Args:
            regressor: The LAKER model to train.
            x_train: Training locations ``(n_train, dx)``.
            y_train: Training targets ``(n_train,)``.
            x_val: Validation locations ``(n_val, dx)``.
            y_val: Validation targets ``(n_val,)``.
            hyperparameters: List of tensors to optimise. If ``None``, defaults
                to ``[lambda_reg_logit]`` (a learned logit for ``lambda_reg``).

        Returns:
            ``self`` for method chaining.

        """
        if x_train.dim() != 2:
            raise ValueError(f"x_train must be 2-D, got shape {x_train.shape}")
        if y_train.dim() != 1:
            raise ValueError(f"y_train must be 1-D, got shape {y_train.shape}")
        if x_val.dim() != 2:
            raise ValueError(f"x_val must be 2-D, got shape {x_val.shape}")
        if y_val.dim() != 1:
            raise ValueError(f"y_val must be 1-D, got shape {y_val.shape}")

        # Default hyperparameter: a learnable logit for lambda_reg
        if hyperparameters is None:
            lambda_logit = torch.tensor(
                [math.log(regressor.lambda_reg)],
                device=self.core.device,
                dtype=self.core.dtype,
                requires_grad=True,
            )
            hyperparameters = [lambda_logit]

        optimizer = torch.optim.Adam(hyperparameters, lr=self.lr)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # ---- inner solve (detach alpha) --------------------------------
            embeddings, model = self.core.compute_embeddings(x_train)
            regressor.embedding_model = model

            # If we are optimising embedding weights, ensure they require grad
            hyper_ids = {id(h) for h in hyperparameters}
            for p in model.parameters():
                if id(p) in hyper_ids:
                    p.requires_grad = True

            kernel_op = self.core.build_kernel_operator(embeddings)
            precond = self.core.build_preconditioner(
                kernel_op.matvec,
                embeddings.shape[0],
                diagonal=kernel_op.diagonal(),
            )
            alpha, _ = self.core.solve_pcg(kernel_op, precond, y_train)
            alpha_detached = alpha.detach()

            # ---- outer loss on validation set ------------------------------
            with torch.no_grad():
                val_embeddings, _ = self.core.compute_embeddings(x_val)
                val_embeddings = (
                    val_embeddings[0] if isinstance(val_embeddings, tuple) else val_embeddings
                )
            k_val = kernel_op.kernel_eval(val_embeddings, embeddings)
            y_val_pred = k_val @ alpha_detached
            val_loss = torch.mean((y_val_pred - y_val) ** 2)

            # ---- hypergradient via implicit differentiation ------------------
            # Analytical gradient of MSE w.r.t. alpha:
            # dL/dalpha = (2/n) * K_val^T @ (K_val @ alpha - y_val)
            n_val = y_val.shape[0]
            residual_val = y_val_pred - y_val
            dL_dalpha = (2.0 / n_val) * (k_val.T @ residual_val)

            hypergrads = implicit_hypergradient(
                operator_fn=kernel_op.matvec,
                preconditioner_fn=precond.apply,
                alpha=alpha_detached,
                dL_dalpha=dL_dalpha,
                param_list=hyperparameters,
                pcg_tol=self.pcg_tol,
                pcg_max_iter=self.pcg_max_iter,
                verbose=False,
            )

            # Apply hypergradients manually (Adam doesn't know about them)
            for param, hg in zip(hyperparameters, hypergrads):
                if param.grad is None:
                    param.grad = hg
                else:
                    param.grad.add_(hg)

            optimizer.step()

            val_loss_item = val_loss.item()
            if self.verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    "Bilevel epoch %d/%d, val_loss=%.4e",
                    epoch + 1,
                    self.epochs,
                    val_loss_item,
                )

            if val_loss_item < best_val_loss:
                best_val_loss = val_loss_item
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        logger.info(
                            "Bilevel early stopping at epoch %d (val_loss=%.4e)",
                            epoch + 1,
                            val_loss_item,
                        )
                    break

        # Final fit with best hyperparameters on full data
        if self.verbose:
            logger.info("Bilevel complete. Refitting on full training set.")
        return regressor.fit(x_train, y_train)
