"""Streaming updates and regularisation-path fitting.

This module provides utilities for incrementally updating a fitted LAKER
model and for fitting a regularisation path over a sequence of
:math:`\\lambda` values.

**Online (partial) updates** (:meth:`StreamingUpdater.partial_fit`).
When new data arrives, the model can be updated without a full refit.
New embeddings are computed, appended to the existing embedding matrix,
and the PCG system is re-solved with a warm start initialised from the
previous solution scaled by a forgetting factor
:math:`\\eta \\in [0, 1]`.  A configurable threshold triggers a full
refit when too many incremental updates have accumulated, preventing
the embedding matrix from growing without bound.

**Regularisation path** (:meth:`StreamingUpdater.fit_path`). Solves
the kernel regression system for a sequence of decreasing
:math:`\\lambda` values, warm-starting each solve from the previous
solution. The path is solved from largest (most regularised) to
smallest (least regularised), which empirically yields faster
convergence since early solves are better conditioned.

**Continuation schedule** (:meth:`StreamingUpdater.fit_continuation`).
A convenience wrapper around :meth:`fit_path` that automatically
constructs a geometrically spaced schedule from
:math:`\\lambda_{\\mathrm{max}}` down to
:math:`\\lambda_{\\mathrm{min}}` over a given number of stages.
The final model state uses :math:`\\lambda_{\\mathrm{min}}`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from laker.core import LAKERCore
    from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class StreamingUpdater:
    """Incremental updates and continuation-path fitting.

    Provides three modes of operation:

    * :meth:`partial_fit` — append new observations and re-solve with
      a warm start, optionally applying a forgetting factor to the
      previous solution.
    * :meth:`fit_path` — solve the kernel regression system over a
      sequence of :math:`\\lambda` values with warm-starting between
      successive solves.
    * :meth:`fit_continuation` — automatically generate a geometrically
      spaced :math:`\\lambda` schedule and delegate to :meth:`fit_path`.

    Args:
        core: The :class:`LAKERCore` instance providing kernel-operator
            construction, preconditioner building, and PCG solving.

    """

    def __init__(self, core: "LAKERCore") -> None:
        """Initialise the streaming updater.

        Args:
            core: :class:`LAKERCore` instance used to build kernel
                operators, preconditioners, and run PCG solves.

        """
        self.core = core

    def partial_fit(
        self,
        regressor: "LAKERRegressor",
        x_new: torch.Tensor,
        y_new: torch.Tensor,
        forgetting_factor: float = 1.0,
        rebuild_threshold: int = 100,
    ) -> "LAKERRegressor":
        """Update the model with one or more new observations.

        Appends the new data to the existing training set and re-solves
        the kernel regression system using a warm start. The previous
        solution vector :math:`\\alpha` is scaled by ``forgetting_factor``
        before being used as the initial guess, which effectively
        discounts older observations.

        After a configurable number of incremental updates (controlled by
        ``rebuild_threshold``), a :class:`RuntimeError` is raised to
        signal that a full refit is required. This prevents the
        embedding matrix from growing indefinitely and ensures the
        preconditioner is periodically rebuilt.

        Args:
            regressor: The fitted :class:`LAKERRegressor` to update.
                Must have been previously fitted via :meth:`fit`.
            x_new: New input locations of shape ``(m, dx)``.
            y_new: New targets of shape ``(m,)``.
            forgetting_factor: Scalar in ``[0, 1]`` that scales the
                previous ``alpha`` before warm-starting. A value of
                ``1.0`` means no forgetting; ``0.0`` discards all prior
                information. Default ``1.0``.
            rebuild_threshold: Maximum cumulative number of new points
                (across all ``partial_fit`` calls) before a full refit is
                required. Default ``100``.

        Returns:
            The updated ``regressor`` with new embeddings, coefficients,
            and preconditioner.

        Raises:
            RuntimeError: If the model has not been fitted, or if
                ``rebuild_threshold`` is exceeded.
            ValueError: If ``x_new`` is not 2-D or ``y_new`` is not 1-D.

        """
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

        kernel_operator = self.core.build_kernel_operator(regressor.embeddings)
        regressor.kernel_operator = kernel_operator

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

        preconditioner = self.core.build_preconditioner(
            kernel_operator.matvec,
            regressor.embeddings.shape[0],
            diagonal=kernel_operator.diagonal(),
        )
        regressor.preconditioner = preconditioner

        regressor.alpha, regressor.pcg_iterations_ = self.core.solve_pcg(
            kernel_operator,
            preconditioner,
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
        """Fit a regularization path over a sequence of lambda_reg values.

        Solves the kernel regression system for each value in
        ``lambda_reg_grid``, sorted from largest to smallest. Each solve
        is warm-started from the previous solution, which typically
        reduces PCG iterations significantly compared to solving from
        scratch.

        The preconditioner can optionally be reused across solves (when
        ``reuse_precond=True``), avoiding the cost of rebuilding it for
        each :math:`\\lambda` value. Since the preconditioner is
        constructed from the kernel operator which does depend on
        :math:`\\lambda`, reuse is an approximation that is most accurate
        when adjacent :math:`\\lambda` values are close.

        The final model state (embeddings, alpha, preconditioner) is set
        to the result for the smallest :math:`\\lambda` in the grid.

        Args:
            regressor: The :class:`LAKERRegressor` to train.
            x: Full input data of shape ``(n, dx)``.
            y: Full target data of shape ``(n,)``.
            lambda_reg_grid: List of regularisation strengths to solve
                for. Processed from largest to smallest.
            reuse_precond: If ``True``, reuse the preconditioner from the
                first solve for all subsequent :math:`\\lambda` values.
                Default ``True``.

        Returns:
            Dictionary with keys:

            * ``"lambda_reg"`` — sorted list of :math:`\\lambda` values.
            * ``"alphas"`` — list of solution vectors, one per
              :math:`\\lambda`.
            * ``"pcg_iters"`` — list of PCG iteration counts.
            * ``"final_rel_res"`` — list of final relative residuals
              :math:`\\|A\\alpha - y\\| / \\|y\\|`.

        Raises:
            ValueError: If ``x`` is not 2-D, ``y`` is not 1-D, or
                ``lambda_reg_grid`` is empty.

        """
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
        kernel_operator = self.core.build_kernel_operator(
            embeddings, lambda_reg=sorted_lambdas[-1]
        )
        regressor.kernel_operator = kernel_operator
        regressor.preconditioner = self.core.build_preconditioner(
            kernel_operator.matvec,
            n,
            gamma=self.core.gamma,
            num_probes=self.core.num_probes,
            diagonal=kernel_operator.diagonal(),
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
        """Fit with a continuation schedule over decreasing regularisation.

        Automatically constructs a geometrically spaced schedule from
        ``lambda_max`` down to ``lambda_min`` over ``n_stages`` stages,
        then delegates to :meth:`fit_path`. The geometric spacing
        ensures each successive :math:`\\lambda` ratio is constant:

        .. math::
            \\lambda_k = \\lambda_{\\mathrm{max}} \\cdot
            r^k, \\quad
            r = \\left(
                \\frac{\\lambda_{\\mathrm{min}}}{\\lambda_{\\mathrm{max}}}
            \\right)^{1/(n-1)}

        Continuation is beneficial because solving from a heavily
        regularised system first provides a good warm start for
        subsequent less-regularised solves, reducing total PCG
        iterations.

        Args:
            regressor: The :class:`LAKERRegressor` to train.
            x: Full input data of shape ``(n, dx)``.
            y: Full target data of shape ``(n,)``.
            lambda_max: Largest regularisation value. If ``None``,
                defaults to ``10 * core.lambda_reg``.
            lambda_min: Smallest regularisation value (the final model
                uses this). If ``None``, defaults to
                ``core.lambda_reg``.
            n_stages: Number of geometrically spaced schedule points.
                Default ``5``.
            reuse_precond: If ``True``, reuse the preconditioner across
                all schedule stages. Default ``True``.

        Returns:
            The fitted ``regressor`` with ``lambda_reg`` set to
            ``lambda_min``.

        Raises:
            ValueError: If ``n_stages`` is not positive or
                ``lambda_max`` / ``lambda_min`` are non-positive.

        """
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
