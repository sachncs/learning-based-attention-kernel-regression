"""End-to-end embedding training and residual correction."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import torch

from laker.bilevel import BilevelOptimizer
from laker.correctors import ResidualCorrector

if TYPE_CHECKING:
    from laker.core import LAKERCore
    from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    """Trains embedding models and residual correctors."""

    def __init__(self, core: "LAKERCore") -> None:
        """Initialise the embedding trainer."""
        self.core = core

    def fit_learned_embeddings(
        self,
        regressor: "LAKERRegressor",
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float = 1e-3,
        epochs: int = 50,
        rebuild_freq: int = 10,
        patience: int = 5,
    ) -> "LAKERRegressor":
        """Optimise the embedding MLP weights end-to-end on the regression objective."""
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        if regressor.embedding_module is None and not hasattr(regressor, "embedding_model"):
            raise RuntimeError(
                "fit_learned_embeddings requires an embedding model. "
                "Call fit() first or pass embedding_module to __init__."
            )

        model = regressor.embedding_model
        if model is None:
            raise RuntimeError("No embedding model available.")

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            for p in model.parameters():
                p.requires_grad = True
            trainable = list(model.parameters())
            if not trainable:
                raise RuntimeError("Embedding model has no trainable parameters.")

        optimizer = torch.optim.Adam(trainable, lr=lr)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            optimizer.zero_grad()

            embedded_input = x.to(dtype=self.core.embedding_dtype)
            embeddings = model(embedded_input)
            if self.core.embedding_dtype != self.core.dtype:
                embeddings = embeddings.to(dtype=self.core.dtype)

            kernel_op = self.core.build_kernel_operator(embeddings)

            if epoch % rebuild_freq == 0 or getattr(regressor, "alpha", None) is None:
                with torch.no_grad():
                    kernel_op_detached = self.core.build_kernel_operator(embeddings.detach())
                    precond = self.core.build_preconditioner(
                        kernel_op_detached.matvec, embeddings.shape[0]
                    )
                    alpha = self.core.solve_pcg(kernel_op_detached, precond, y)[0]
                regressor.preconditioner = precond
            else:
                alpha = regressor.alpha.detach()

            residual = kernel_op.matvec(alpha) - y
            loss = 0.5 * torch.dot(residual, residual)

            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            if self.core.verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "Learned embeddings epoch %d/%d, loss=%.4e",
                    epoch + 1,
                    epochs,
                    loss_item,
                )

            if loss_item < best_loss:
                best_loss = loss_item
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.core.verbose:
                        logger.info(
                            "Early stopping at epoch %d (loss=%.4e)",
                            epoch + 1,
                            loss_item,
                        )
                    break

        regressor.embeddings = embeddings.detach()
        regressor.kernel_operator = kernel_op
        regressor.alpha, regressor.pcg_iterations_ = self.core.solve_pcg(
            regressor.kernel_operator, regressor.preconditioner, y
        )
        return regressor

    def fit_residual_corrector(
        self,
        regressor: "LAKERRegressor",
        x: torch.Tensor,
        y: torch.Tensor,
        val_fraction: float = 0.2,
        epochs: int = 200,
        patience: int = 10,
        weight_decay: float = 1e-2,
        lr: float = 1e-3,
    ) -> "LAKERRegressor":
        """Train a small residual corrector on ``y - y_hat_laker``."""
        if regressor.alpha is None or regressor.embeddings is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before fit_residual_corrector()."
            )

        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        n = x.shape[0]
        n_val = max(1, int(n * val_fraction))
        indices = torch.randperm(n, device=x.device)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]

        with torch.no_grad():
            y_train_base = regressor.predict(x_train).detach()
            y_val_base = regressor.predict(x_val).detach()

        residuals_train = y_train - y_train_base
        residuals_val = y_val - y_val_base

        input_dim = x.shape[1]
        if regressor.residual_corrector is None:
            regressor.residual_corrector = ResidualCorrector(
                input_dim=input_dim,
                output_dim=1,
                hidden_dim=32,
                dropout=0.1,
            ).to(self.core.device)
        else:
            regressor.residual_corrector = regressor.residual_corrector.to(self.core.device)

        optimizer = torch.optim.Adam(
            regressor.residual_corrector.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[dict] = None

        for epoch in range(epochs):
            regressor.residual_corrector.train()
            optimizer.zero_grad()
            pred = regressor.residual_corrector(x_train).squeeze()
            loss = torch.mean((pred - residuals_train) ** 2)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                regressor.residual_corrector.eval()
                val_pred = regressor.residual_corrector(x_val).squeeze()
                val_loss = torch.mean((val_pred - residuals_val) ** 2).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in regressor.residual_corrector.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.core.verbose:
                        logger.info(
                            "Residual corrector early stopping at epoch %d (val_loss=%.4e)",
                            epoch + 1,
                            val_loss,
                        )
                    break

        if best_state is not None:
            regressor.residual_corrector.load_state_dict(best_state)

        if self.core.verbose:
            logger.info(
                "Residual corrector fitted: epochs=%d, best_val_loss=%.4e",
                epoch + 1,
                best_val_loss,
            )
        return regressor

    def fit_bilevel(
        self,
        regressor: "LAKERRegressor",
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        lr: float = 1e-3,
        epochs: int = 20,
        patience: int = 5,
    ) -> "LAKERRegressor":
        """Optimise hyperparameters via bilevel learning with implicit differentiation."""
        bilevel = BilevelOptimizer(
            core=self.core,
            lr=lr,
            epochs=epochs,
            patience=patience,
            verbose=self.core.verbose,
        )
        return bilevel.fit_bilevel(regressor, x_train, y_train, x_val, y_val)

    def fit_uncertainty_aware(
        self,
        regressor: "LAKERRegressor",
        x: torch.Tensor,
        y: torch.Tensor,
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
        For the RFF kernel the exact closed-form variance is used; for other
        kernels a differentiable distance-to-manifold proxy is used.
        """
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        if regressor.embedding_model is None:
            raise RuntimeError(
                "fit_uncertainty_aware requires an embedding model. "
                "Call fit() first or pass embedding_module to __init__."
            )

        model = regressor.embedding_model
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            for p in model.parameters():
                p.requires_grad = True
            trainable = list(model.parameters())
            if not trainable:
                raise RuntimeError("Embedding model has no trainable parameters.")

        optimizer = torch.optim.Adam(trainable, lr=lr)
        n = x.shape[0]
        n_subset = max(1, int(n * variance_subset))

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            optimizer.zero_grad()

            embedded_input = x.to(dtype=self.core.embedding_dtype)
            embeddings = model(embedded_input)
            if self.core.embedding_dtype != self.core.dtype:
                embeddings = embeddings.to(dtype=self.core.dtype)

            kernel_op = self.core.build_kernel_operator(embeddings)
            with torch.no_grad():
                kernel_op_detached = self.core.build_kernel_operator(embeddings.detach())
                precond = self.core.build_preconditioner(
                    kernel_op_detached.matvec,
                    n,
                    diagonal=kernel_op_detached.diagonal(),
                )
                alpha, _ = self.core.solve_pcg(kernel_op_detached, precond, y)

            # Differentiable prediction on full training set
            mu = self.core.predict_train(
                x,
                model,
                embeddings,
                kernel_op,
                alpha,
                regressor.residual_corrector,
            )

            # Variance on a random subset (stochastic approximation)
            perm = torch.randperm(n, device=x.device)
            subset_idx = perm[:n_subset]
            x_subset = x[subset_idx]
            var = self.core.predict_variance_train(
                x_subset,
                model,
                embeddings,
                kernel_op,
                precond,
                alpha,
                self.core.lambda_reg,
            )

            residual = y[subset_idx] - mu[subset_idx]
            nll = 0.5 * torch.mean(torch.log(2.0 * math.pi * var) + (residual**2) / var)
            calibration = (torch.mean(residual**2) - torch.mean(var)) ** 2
            loss = nll + beta * calibration

            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            if self.core.verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "Uncertainty-aware epoch %d/%d, loss=%.4e (nll=%.4e, cal=%.4e)",
                    epoch + 1,
                    epochs,
                    loss_item,
                    nll.item(),
                    calibration.item(),
                )

            if loss_item < best_loss:
                best_loss = loss_item
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.core.verbose:
                        logger.info(
                            "Uncertainty-aware early stopping at epoch %d (loss=%.4e)",
                            epoch + 1,
                            loss_item,
                        )
                    break

        # Final fit with converged embeddings on full data
        regressor.embeddings = embeddings.detach()
        regressor.kernel_operator = self.core.build_kernel_operator(regressor.embeddings)
        regressor.preconditioner = self.core.build_preconditioner(
            regressor.kernel_operator.matvec,
            n,
            diagonal=regressor.kernel_operator.diagonal(),
        )
        regressor.alpha, regressor.pcg_iterations_ = self.core.solve_pcg(
            regressor.kernel_operator, regressor.preconditioner, y
        )
        return regressor
