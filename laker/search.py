"""Hyperparameter search: grid search and Bayesian optimisation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy
import torch

if TYPE_CHECKING:
    from laker.core import LAKERCore
    from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class HyperparameterSearch:
    """Validation-based hyperparameter search for LAKER."""

    def __init__(self, core: "LAKERCore") -> None:
        """Initialise the hyperparameter search."""
        self.core = core

    def fit_with_search(
        self,
        regressor: "LAKERRegressor",
        x: torch.Tensor,
        y: torch.Tensor,
        val_fraction: float = 0.2,
        lambda_reg_grid: Optional[list[float]] = None,
        gamma_grid: Optional[list[float]] = None,
        num_probes_grid: Optional[list[int]] = None,
        warm_start: bool = True,
    ) -> "LAKERRegressor":
        """Fit with validation-based grid search over key hyperparameters."""
        n = x.shape[0]
        n_val = max(1, int(n * val_fraction))
        indices = torch.randperm(n, device=x.device)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        if lambda_reg_grid is None:
            lambda_reg_grid = [1e-3, 1e-2, 1e-1]
        if gamma_grid is None:
            gamma_grid = [0.0, 1e-1, 1.0]
        if num_probes_grid is None:
            num_probes_grid = [50, 100, 200]

        full_embeddings, model = self.core.compute_embeddings(x)
        regressor.embedding_model = model
        train_embeddings = full_embeddings[train_idx]

        best_rmse = float("inf")
        best_params = {}
        best_alpha: Optional[torch.Tensor] = None

        if self.core.verbose:
            logger.info(
                "Starting grid search: %d lambda x %d gamma x %d probes = %d configs",
                len(lambda_reg_grid),
                len(gamma_grid),
                len(num_probes_grid),
                len(lambda_reg_grid) * len(gamma_grid) * len(num_probes_grid),
            )

        x0 = None
        for lambda_reg_value in lambda_reg_grid:
            for gamma_value in gamma_grid:
                for num_probes_value in num_probes_grid:
                    try:
                        kernel_op = self.core.build_kernel_operator(
                            train_embeddings, lambda_reg=lambda_reg_value
                        )
                        precond = self.core.build_preconditioner(
                            kernel_op.matvec,
                            train_embeddings.shape[0],
                            gamma=gamma_value,
                            num_probes=num_probes_value,
                            diagonal=kernel_op.diagonal(),
                        )
                        alpha, _ = self.core.solve_pcg(
                            kernel_op,
                            precond,
                            y[train_idx],
                            x0=x0 if warm_start else None,
                        )
                        x_val_embed = x[val_idx].to(dtype=self.core.embedding_dtype)
                        with torch.no_grad():
                            val_embeddings = regressor.embedding_model(x_val_embed)
                        if self.core.embedding_dtype != self.core.dtype:
                            val_embeddings = val_embeddings.to(dtype=self.core.dtype)
                        k_val = kernel_op.kernel_eval(val_embeddings, train_embeddings)
                        y_val_pred = k_val @ alpha
                        rmse = torch.sqrt(torch.mean((y_val_pred - y[val_idx]) ** 2)).item()
                    except (RuntimeError, ValueError) as exc:
                        rmse = float("inf")
                        if self.core.verbose:
                            logger.warning(
                                "Trial failed: lambda=%.3e gamma=%.3e probes=%d (%s)",
                                lambda_reg_value,
                                gamma_value,
                                num_probes_value,
                                exc,
                            )

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {
                            "lambda_reg": lambda_reg_value,
                            "gamma": gamma_value,
                            "num_probes": num_probes_value,
                        }
                        best_alpha = alpha
                        if self.core.verbose:
                            logger.info(
                                "New best: lambda=%.3e gamma=%.3e probes=%d val_rmse=%.4f",
                                lambda_reg_value,
                                gamma_value,
                                num_probes_value,
                                rmse,
                            )

                    if warm_start and best_alpha is not None:
                        x0 = best_alpha.clone()

        if not best_params:
            raise RuntimeError(
                "Grid search failed: all parameter combinations diverged or raised errors. "
                "Try widening lambda_reg_grid, increasing pcg_max_iter, "
                "or using dtype=torch.float64."
            )

        if self.core.verbose:
            logger.info(
                "Best hyperparameters: lambda_reg=%.3e gamma=%.3e num_probes=%d",
                best_params["lambda_reg"],
                best_params["gamma"],
                best_params["num_probes"],
            )

        regressor.lambda_reg = float(best_params["lambda_reg"])
        regressor.gamma = float(best_params["gamma"])
        regressor.num_probes = int(best_params["num_probes"])
        return regressor.fit(x, y)

    def fit_with_bo(
        self,
        regressor: "LAKERRegressor",
        x: torch.Tensor,
        y: torch.Tensor,
        val_fraction: float = 0.2,
        n_calls: int = 15,
        n_initial_points: int = 5,
        lambda_reg_bounds: tuple[float, float] = (1e-4, 1.0),
        gamma_bounds: tuple[float, float] = (0.0, 2.0),
        num_probes_bounds: tuple[int, int] = (20, 300),
    ) -> "LAKERRegressor":
        """Fit with Bayesian Optimisation over key hyperparameters."""
        import numpy as np

        from laker.utils import GPSurrogate

        n = x.shape[0]
        n_val = max(1, int(n * val_fraction))
        indices = torch.randperm(n, device=x.device)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        embeddings, model = self.core.compute_embeddings(x)
        regressor.embedding_model = model
        train_embeddings = embeddings[train_idx]

        bounds = np.array(
            [lambda_reg_bounds, gamma_bounds, num_probes_bounds],
            dtype=np.float64,
        )

        def lh_sample(n_samp: int) -> np.ndarray:
            d = bounds.shape[0]
            samples = np.zeros((n_samp, d), dtype=np.float64)
            for i in range(d):
                perm = np.random.permutation(n_samp)
                samples[:, i] = (perm + np.random.uniform(size=n_samp)) / n_samp
            return samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        X_obs = []
        y_obs = []

        for _ in range(n_initial_points):
            point = lh_sample(1)[0]
            lambda_reg_value, gamma_value, num_probes_value = (
                float(point[0]),
                float(point[1]),
                int(round(float(point[2]))),
            )
            rmse = self.bo_eval(
                regressor,
                train_embeddings,
                x[val_idx],
                y,
                train_idx,
                val_idx,
                lambda_reg_value,
                gamma_value,
                num_probes_value,
            )
            X_obs.append(point)
            y_obs.append(rmse)

        gp = GPSurrogate(bounds, log_indices=[0, 1])
        best_rmse = min(y_obs)
        best_arr = X_obs[numpy.argmin(y_obs)]
        best_params = {
            "lambda_reg": float(best_arr[0]),
            "gamma": float(best_arr[1]),
            "num_probes": int(round(float(best_arr[2]))),
        }

        for _ in range(n_calls - n_initial_points):
            gp.fit(numpy.vstack(X_obs), numpy.array(y_obs, dtype=numpy.float64))
            candidates = lh_sample(500)
            ei = gp.expected_improvement(candidates)
            next_point = candidates[numpy.argmax(ei)]

            lambda_reg_value, gamma_value, num_probes_value = (
                next_point[0],
                next_point[1],
                int(round(next_point[2])),
            )
            rmse = self.bo_eval(
                regressor,
                train_embeddings,
                x[val_idx],
                y,
                train_idx,
                val_idx,
                lambda_reg_value,
                gamma_value,
                num_probes_value,
            )
            X_obs.append(next_point)
            y_obs.append(rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    "lambda_reg": lambda_reg_value,
                    "gamma": gamma_value,
                    "num_probes": num_probes_value,
                }

        if self.core.verbose:
            logger.info(
                "Best BO hyperparameters: lambda_reg=%.3e gamma=%.3e num_probes=%d",
                best_params["lambda_reg"],
                best_params["gamma"],
                best_params["num_probes"],
            )

        regressor.lambda_reg = float(best_params["lambda_reg"])
        regressor.gamma = float(best_params["gamma"])
        regressor.num_probes = int(best_params["num_probes"])
        return regressor.fit(x, y)

    def bo_eval(
        self,
        regressor: "LAKERRegressor",
        train_embeddings: torch.Tensor,
        x_val: torch.Tensor,
        y: torch.Tensor,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        lambda_reg_value: float,
        gamma_value: float,
        num_probes_value: int,
    ) -> float:
        """Single BO evaluation: fit on train, predict on val, return RMSE."""
        try:
            kernel_op = self.core.build_kernel_operator(
                train_embeddings, lambda_reg=lambda_reg_value
            )
            precond = self.core.build_preconditioner(
                kernel_op.matvec,
                train_embeddings.shape[0],
                gamma=gamma_value,
                num_probes=num_probes_value,
                diagonal=kernel_op.diagonal(),
            )
            alpha, _ = self.core.solve_pcg(kernel_op, precond, y[train_idx])

            x_val_embed = x_val.to(dtype=self.core.embedding_dtype)
            with torch.no_grad():
                val_embeddings = regressor.embedding_model(x_val_embed)
            if self.core.embedding_dtype != self.core.dtype:
                val_embeddings = val_embeddings.to(dtype=self.core.dtype)
            k_val = kernel_op.kernel_eval(val_embeddings, train_embeddings)
            y_val_pred = k_val @ alpha
            return torch.sqrt(torch.mean((y_val_pred - y[val_idx]) ** 2)).item()
        except (RuntimeError, ValueError):
            logger.debug("bo_eval failed", exc_info=True)
            return float("inf")
