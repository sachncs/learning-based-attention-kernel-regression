"""End-to-end embedding training, residual correction, and bilevel learning.

This module provides the :class:`EmbeddingTrainer` class, which implements
several training strategies for the LAKER pipeline:

**End-to-end embedding optimisation** (:meth:`EmbeddingTrainer.fit_learned_embeddings`):
    Gradient-based optimisation of a neural embedding MLP on the regression
    objective :math:`\\frac{1}{2}\\|K(E)\\alpha - y\\|_2^2`.  The kernel
    matrix :math:`K(E) = \\exp(E E^\\top)` depends on the learned embedding
    :math:`E`, enabling back-propagation through the attention kernel.  A
    full PCG re-solve is performed every ``rebuild_freq`` epochs to keep the
    weight vector :math:`\\alpha` on the normal equation manifold.

**Residual correction** (:meth:`EmbeddingTrainer.fit_residual_corrector`):
    Trains a small auxiliary MLP on the residual
    :math:`r = y - \\hat{y}_{\\text{laker}}` after the base LAKER fit.
    The corrector prediction is added to the base output to compensate for
    local model misspecification without destabilising the kernel solver.

**Bilevel hyperparameter learning** (:meth:`EmbeddingTrainer.fit_bilevel`):
    Uses implicit differentiation through the PCG fixed-point to compute
    hypergradients for continuous hyperparameters (e.g. ``lambda_reg``,
    embedding weights).  An outer-loop Adam optimiser minimises validation
    loss while the inner solve is held approximately at stationarity.

**Uncertainty-aware training** (:meth:`EmbeddingTrainer.fit_uncertainty_aware`):
    Jointly optimises embeddings on a negative log-likelihood (NLL) plus a
    calibration penalty:

    .. math::

        \\mathcal{L} = \\text{NLL}(y \\mid \\mu, \\sigma^2)
        + \\beta \\,(\\mathbb{E}[r^2] - \\mathbb{E}[\\sigma^2])^2

    where :math:`\\mu` and :math:`\\sigma^2` are the LAKER predictive mean
    and variance respectively.  This prevents overconfident predictions and
    improves uncertainty quantification for downstream tasks such as active
    sensing.

All training methods follow the same pattern:

1. Validate input shapes and model state.
2. Run an optimisation loop with early stopping.
3. Update the fitted state on the :class:`~laker.models.LAKERRegressor`
   (embeddings, kernel operator, preconditioner, alpha).
"""

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
    """Coordinates embedding and residual-corrector training for LAKER.

    The trainer holds a reference to the :class:`~laker.core.LAKERCore`
    pipeline instance which provides the kernel builder, preconditioner,
    and PCG solver.  It does **not** own any trainable state itself;
    instead it mutates the fitted attributes on the
    :class:`~laker.models.LAKERRegressor` that is passed to each method.

    Attributes:
        core: The :class:`~laker.core.LAKERCore` pipeline providing kernel
            construction, preconditioning, and solve routines.

    Args:
        core: The LAKER pipeline core instance.
    """

    def __init__(self, core: "LAKERCore") -> None:
        """Initialise the embedding trainer.

        Args:
            core: The :class:`~laker.core.LAKERCore` pipeline instance that
                provides kernel construction, preconditioner building, and
                PCG solve routines used throughout training.
        """
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
        """Optimise the embedding MLP weights end-to-end on the regression objective.

        Performs gradient-based training of the neural embedding network by
        minimising the least-squares loss

        .. math::

            \\mathcal{L}(\\theta) = \\frac{1}{2}\\|K(\\theta)\\alpha - y\\|_2^2

        where :math:`K(\\theta) = \\exp(E(\\theta) E(\\theta)^\\top)` is the
        attention kernel parameterised by the embedding weights :math:`\\theta`.
        Every ``rebuild_freq`` epochs the preconditioner is recomputed and
        :math:`\\alpha` is re-solved via PCG to keep the solution on the
        normal equation manifold.

        Args:
            regressor: The :class:`~laker.models.LAKERRegressor` whose
                ``embedding_model`` will be optimised.  The regressor must
                have been fitted at least once (so that ``embedding_model``
                is not ``None``).
            x: Training locations of shape ``(n, d)`` where ``n`` is the
                number of samples and ``d`` the spatial dimension.
            y: Training observations of shape ``(n,)``.
            lr: Learning rate for the Adam optimiser.
            epochs: Maximum number of optimisation epochs.
            rebuild_freq: Frequency (in epochs) at which the preconditioner
                is rebuilt and ``alpha`` is re-solved.
            patience: Number of epochs without improvement before early
                stopping.

        Returns:
            The same ``regressor`` instance with updated ``embeddings``,
            ``alpha``, ``kernel_operator``, and ``preconditioner`` attributes.

        Raises:
            ValueError: If ``x`` is not 2-D or ``y`` is not 1-D.
            RuntimeError: If the regressor has no embedding model or no
                trainable parameters.

        Examples:
            >>> from laker.models import LAKERRegressor
            >>> reg = LAKERRegressor(embedding_dim=16, kernel_approx="rff",
            ...                      num_features=128)
            >>> reg.fit(x_train, y_train)
            >>> reg.fit_learned_embeddings(x_train, y_train, lr=5e-4,
            ...                           epochs=100)
        """
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
        precond = regressor.preconditioner
        assert precond is not None
        regressor.alpha, regressor.pcg_iterations_ = self.core.solve_pcg(
            kernel_op, precond, y
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
        """Train a small residual corrector on ``y - y_hat_laker``.

        After the base LAKER fit, a lightweight MLP corrector is trained to
        predict the residual :math:`r = y - \\hat{y}_{\\text{laker}}` from the
        raw spatial coordinates.  The corrector prediction is added to the base
        LAKER output at inference time, compensating for local model
        misspecification without destabilising the kernel solver.

        Training uses a held-out validation split (``val_fraction`` of the
        training data) for early stopping.  The best model state (by
        validation loss) is restored after training.

        Args:
            regressor: The :class:`~laker.models.LAKERRegressor` whose base
                fit will be refined.  Must already be fitted (``alpha`` and
                ``embeddings`` must not be ``None``).
            x: Training locations of shape ``(n, d)``.
            y: Training observations of shape ``(n,)``.
            val_fraction: Fraction of the training data held out for
                validation and early stopping.  Must be in ``(0, 1)``.
            epochs: Maximum number of training epochs.
            patience: Number of consecutive validation epochs without
                improvement before early stopping.
            weight_decay: L2 regularisation coefficient for the Adam
                optimiser.
            lr: Learning rate for the Adam optimiser.

        Returns:
            The same ``regressor`` instance with ``residual_corrector``
            populated or updated.

        Raises:
            ValueError: If ``x`` is not 2-D or ``y`` is not 1-D.
            RuntimeError: If the model has not been fitted yet.

        Examples:
            >>> reg = LAKERRegressor()
            >>> reg.fit(x_train, y_train)
            >>> reg.fit_residual_corrector(x_train, y_train, val_fraction=0.3)
        """
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
        """Optimise hyperparameters via bilevel learning with implicit differentiation.

        Formulates hyperparameter optimisation as a bilevel problem:

        .. math::

            \\min_{\\theta} \\; \\mathcal{L}_{\\text{val}}(\\alpha^*(\\theta), \\theta)
            \\quad \\text{s.t.} \\quad (K(\\theta) + \\lambda I)\\alpha^*(\\theta) = y

        where the inner problem (PCG solve) is differentiated through using
        implicit differentiation (Neumann-series approximation of the
        hypergradient).  An outer-loop Adam optimiser updates the
        hyperparameters :math:`\\theta` (e.g. ``lambda_reg``, embedding
        weights) to minimise validation loss.

        Delegates the actual computation to
        :class:`~laker.bilevel.BilevelOptimizer`.

        Args:
            regressor: The :class:`~laker.models.LAKERRegressor` to optimise.
            x_train: Training locations of shape ``(n, d)``.
            y_train: Training observations of shape ``(n,)``.
            x_val: Validation locations of shape ``(m, d)``.
            y_val: Validation observations of shape ``(m,)``.
            lr: Learning rate for the outer-loop Adam optimiser.
            epochs: Maximum number of outer-loop epochs.
            patience: Early-stopping patience on validation loss.

        Returns:
            The same ``regressor`` instance with updated hyperparameters
            and fitted state.

        Examples:
            >>> reg = LAKERRegressor(lambda_reg=0.1)
            >>> reg.fit(x_train, y_train)
            >>> reg.fit_bilevel(x_train, y_train, x_val, y_val, epochs=30)
        """
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

        Jointly optimises the embedding MLP and predicts well-calibrated
        uncertainty estimates.  The loss is:

        .. math::

            \\mathcal{L} = \\text{NLL}(y \\mid \\mu, \\sigma^2)
            + \\beta \\,(\\mathbb{E}[r^2] - \\mathbb{E}[\\sigma^2])^2

        where :math:`\\mu` is the LAKER predictive mean, :math:`\\sigma^2`
        is the predictive variance, and :math:`r = y - \\mu`.  For the RFF
        kernel the exact closed-form variance is used; for other kernels a
        differentiable distance-to-manifold proxy is employed.  The variance
        is computed on a random subset (``variance_subset`` fraction of the
        training data) as a stochastic approximation.

        After convergence the embeddings are frozen and a final PCG solve is
        performed on the full training data.

        Args:
            regressor: The :class:`~laker.models.LAKERRegressor` whose
                ``embedding_model`` will be optimised.  Must have been
                fitted at least once.
            x: Training locations of shape ``(n, d)``.
            y: Training observations of shape ``(n,)``.
            lr: Learning rate for the Adam optimiser.
            epochs: Maximum number of optimisation epochs.
            beta: Weight of the calibration penalty term.  Higher values
                enforce tighter calibration between residual variance and
                predicted variance.
            variance_subset: Fraction of training points used for the
                stochastic variance estimate each epoch.
            patience: Early-stopping patience on the combined loss.

        Returns:
            The same ``regressor`` instance with updated ``embeddings``,
            ``alpha``, ``kernel_operator``, and ``preconditioner`` attributes.

        Raises:
            ValueError: If ``x`` is not 2-D or ``y`` is not 1-D.
            RuntimeError: If the regressor has no embedding model or no
                trainable parameters.

        Examples:
            >>> reg = LAKERRegressor(embedding_dim=16, kernel_approx="rff",
            ...                      num_features=128)
            >>> reg.fit(x_train, y_train)
            >>> reg.fit_uncertainty_aware(x_train, y_train, beta=0.05)
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
        kernel_operator = self.core.build_kernel_operator(regressor.embeddings)
        regressor.kernel_operator = kernel_operator
        preconditioner = self.core.build_preconditioner(
            kernel_operator.matvec,
            n,
            diagonal=kernel_operator.diagonal(),
        )
        regressor.preconditioner = preconditioner
        regressor.alpha, regressor.pcg_iterations_ = self.core.solve_pcg(
            kernel_operator, preconditioner, y
        )
        return regressor
