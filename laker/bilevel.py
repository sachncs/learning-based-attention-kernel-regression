"""Bilevel hyperparameter learning via implicit differentiation.

Bilevel optimisation treats hyperparameter selection as a nested
optimisation problem:

* **Inner problem.** Given hyperparameters :math:`\\theta` (e.g.
  regularisation strength :math:`\\lambda`, embedding weights), solve
  the kernel regression system

  .. math::
      (K(\\theta) + \\lambda I) \\alpha = y

  via preconditioned conjugate gradient (PCG).

* **Outer problem.** Minimise a validation loss
  :math:`\\mathcal{L}_{\\mathrm{val}}(\\alpha(\\theta))` with respect
  to :math:`\\theta` using an Adam optimiser.

Gradients of the outer loss with respect to :math:`\\theta` are
computed by :mod:`laker.implicit_diff`, which implements the adjoint
method: one additional PCG solve for the adjoint vector, followed by
cheap per-parameter dot products.  This avoids differentiating through
every CG iteration, which would be prohibitively expensive.

The :class:`BilevelOptimizer` class orchestrates this outer loop,
alternating between the inner PCG solve, outer loss evaluation, and
hypergradient computation until convergence or early stopping.
"""

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

    The inner problem solves :math:`(K(\\theta) + \\lambda I) \\alpha = y`
    via preconditioned conjugate gradient (PCG), where :math:`K(\\theta)`
    is a kernel operator whose construction depends on learnable
    hyperparameters :math:`\\theta`.

    The outer problem minimises validation mean-squared error with respect
    to :math:`\\theta` using Adam. Hypergradients are computed by the
    adjoint method (:mod:`laker.implicit_diff`), which requires only one
    additional PCG solve per outer iteration.

    By default the only learnable hyperparameter is a logit for the
    regularisation strength :math:`\\lambda`, but custom parameter lists
    (e.g. embedding-network weights) can be supplied.

    Args:
        core: The :class:`LAKERCore` instance providing kernel-operator
            construction, preconditioner building, and PCG solving.
        lr: Learning rate for the outer Adam optimiser.
        epochs: Maximum number of outer iterations.
        patience: Early-stopping patience (number of outer iterations
            without validation-loss improvement before stopping).
        pcg_tol: Tolerance forwarded to the inner PCG solve.
        pcg_max_iter: Maximum iterations forwarded to the inner PCG solve.
        verbose: Whether to log progress.

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
        """Initialise the bilevel optimiser.

        Args:
            core: :class:`LAKERCore` instance that provides
                :meth:`compute_embeddings`, :meth:`build_kernel_operator`,
                :meth:`build_preconditioner`, and :meth:`solve_pcg`.
            lr: Adam learning rate for the outer optimisation loop.
            epochs: Maximum number of outer iterations.
            patience: Early-stopping patience. Training stops if the
                validation loss has not improved for this many
                consecutive iterations.
            pcg_tol: Relative tolerance for the inner PCG solve.
            pcg_max_iter: Maximum PCG iterations for the inner solve.
            verbose: Whether to log per-epoch progress.

        """
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

        Performs the full outer loop:

        1. Compute embeddings and build the kernel operator from
           ``x_train``.
        2. Solve the inner PCG system for :math:`\\alpha`.
        3. Evaluate the validation MSE on ``x_val`` / ``y_val``.
        4. Compute hypergradients via the adjoint method.
        5. Update hyperparameters with Adam.
        6. Repeat until convergence or early stopping, then refit the
           model on the full training set with the best hyperparameters.

        Args:
            regressor: The :class:`LAKERRegressor` to train. Its
                :meth:`fit` method is called at the end with the full
                training data.
            x_train: Training input locations of shape ``(n_train, dx)``.
            y_train: Training targets of shape ``(n_train,)``.
            x_val: Validation input locations of shape ``(n_val, dx)``.
            y_val: Validation targets of shape ``(n_val,)``.
            hyperparameters: List of tensors to optimise in the outer
                loop. If ``None``, defaults to a single learnable logit
                for ``regressor.lambda_reg``.

        Returns:
            The fitted ``regressor`` (after a final refit on all data).

        Raises:
            ValueError: If input tensors have incorrect dimensionality.

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
